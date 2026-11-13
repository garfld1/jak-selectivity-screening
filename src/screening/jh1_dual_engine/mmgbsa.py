#!/usr/bin/env python3
"""
Single-point MM/GBSA energies for poses the docking engines already kept.

    G = E(complex) - E(receptor) - E(ligand)

Heavy atoms stay at the docked coordinates. Receptor hydrogens are relaxed
once per worker and cached. Ligand hydrogens are relaxed on the free ligand,
which is a small system. The complex energy is then one evaluation in GBn2
at those coordinates, not a minimization of the whole kinase for every pose.

OpenMM is pinned to the CPU platform. Thread count comes from
``OPENMM_CPU_THREADS``, which the docking driver sets to one share of the
machine per worker, so a process pool does not start an OpenCL context or
oversubscribe the CPU.

The receptor includes the explicit site waters written by
``prepare_receptors.py``. Phosphotyrosine is converted to tyrosine only in
this copy, because the activation-loop sites sit well outside the docking
box and ff14SB has no PTR template. Docking and PLIP still see PTR. The
C-terminal residue gets an OXT atom in this copy too. ColabFold models omit
it, and ff14SB will not treat the chain end as terminal without it.

Ligand parameters are OpenFF Sage. Charges come from the production NAGL
AM1-BCC model so AmberTools is not required. More negative G ranks first.
A non-finite pose energy is returned as NaN; the caller drops that pose.
"""

from __future__ import annotations

import collections
import io
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from rdkit import Chem

METHOD_ID = "openmm-gbn2-ff14sb-openff-sage-nagl-sp"
# Finished jobs from the hydrogen-minimized complex protocol. Resume keeps
# their scores instead of docking those compounds again.
PREVIOUS_METHOD_ID = "openmm-gbn2-ff14sb-openff-sage-nagl"

PROTEIN_FF = (
    "amber/protein.ff14SB.xml",
    "amber/tip3p_standard.xml",
    "amber/tip3p_HFE_multivalent.xml",
    "implicit/gbn2.xml",
)
SAGE_PREFERENCE = (
    "openff-2.2.1",
    "openff-2.2.0",
    "openff-2.1.1",
    "openff-2.1.0",
    "openff-2.0.0",
)
NAGL_MODELS = (
    "openff-gnn-am1bcc-1.0.0.pt",
    "openff-gnn-am1bcc-0.1.0-rc.3.pt",
    "openff-gnn-am1bcc-0.1.0-rc.1.pt",
)
MAX_H_STEPS = 100

# One entry per worker process. A worker docks many ligands into the same receptor.
_RECEPTOR_CACHE: Dict[Tuple[str, int, int], "PreparedReceptor"] = {}


def cpu_threads_for_workers(workers: int) -> int:
    """One share of the CPUs per docking worker. OpenMM then stays inside that share."""
    cpus = os.cpu_count() or 1
    return max(1, cpus // max(1, int(workers)))


class PreparedReceptor:
    def __init__(self, topology, positions, energy: float):
        self.topology = topology
        self.positions = positions
        self.energy = energy


def mmgbsa_pose_energies(receptor_pdb: str, pose_mols: Sequence[Chem.Mol]) -> List[float]:
    """MM/GBSA for each pose, in kcal/mol. NaN means that pose could not be scored."""
    if not pose_mols:
        return []
    poses = [_ensure_hydrogens(mol) for mol in pose_mols]
    _require_same_chemistry(poses)

    receptor = _get_receptor(receptor_pdb)
    ligand = _off_molecule(poses[0])
    ligand_topology = _ligand_topology(poses[0])
    forcefield = _forcefield(ligand)
    ligand_system = _build_system(forcefield, ligand_topology)
    ligand_context = _context(ligand_system, ligand_topology)

    complex_topology, _ = _merge(receptor, ligand_topology, _conformer_nm(poses[0]))
    complex_system = _build_system(forcefield, complex_topology)
    complex_context = _context(complex_system, complex_topology, freeze_heavy=False)

    energies: List[float] = []
    for pose in poses:
        ligand_nm = _conformer_nm(pose)
        if ligand_nm.shape[0] != ligand_topology.getNumAtoms():
            raise RuntimeError("pose atom count does not match the ligand used to build the system")
        try:
            ligand_energy, ligand_positions = _minimized_state(ligand_context, _quantity(ligand_nm))
            complex_energy = _potential_energy(
                complex_context, _join_positions(receptor.positions, ligand_positions)
            )
        except Exception:
            energies.append(float("nan"))
            continue
        value = complex_energy - receptor.energy - ligand_energy
        energies.append(value if math.isfinite(value) else float("nan"))
    return energies


_WATER_NAMES = {"HOH", "WAT"}
_OXT_BOND_A = 1.25


def _pdb_xyz(line: str) -> np.ndarray:
    return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)


def _oxt_coordinate(ca: np.ndarray, carbon: np.ndarray, oxygen: np.ndarray) -> np.ndarray:
    """Place OXT in the carboxyl plane, 120 degrees from both CA and O."""
    toward_ca = ca - carbon
    toward_o = oxygen - carbon
    toward_ca /= np.linalg.norm(toward_ca)
    toward_o /= np.linalg.norm(toward_o)
    direction = -(toward_ca + toward_o)
    direction /= np.linalg.norm(direction)
    return carbon + direction * _OXT_BOND_A


def _format_oxt(template: str, serial: int, xyz: np.ndarray) -> str:
    padded = f"{template:<80}"
    x, y, z = (float(value) for value in xyz)
    return (
        f"ATOM  {serial:5d}  OXT{padded[16:30]}"
        f"{x:8.3f}{y:8.3f}{z:8.3f}{padded[54:76]}O "
    )


def _add_c_terminal_oxt(pdb_text: str) -> str:
    """Add a carboxyl OXT to the last protein residue of each chain."""
    lines = pdb_text.splitlines()
    atom_indexes = [i for i, line in enumerate(lines) if line.startswith(("ATOM", "HETATM"))]
    groups: List[Tuple[Tuple[str, str, str], List[int]]] = []
    current_key = None
    current: List[int] = []
    for index in atom_indexes:
        line = lines[index]
        key = (line[21], line[22:27], line[17:20])
        if key != current_key:
            if current:
                groups.append((current_key, current))
            current_key = key
            current = [index]
        else:
            current.append(index)
    if current:
        groups.append((current_key, current))

    last_by_chain: Dict[str, Tuple[Tuple[str, str, str], List[int]]] = {}
    for key, indexes in groups:
        if key[2].strip() in _WATER_NAMES:
            continue
        last_by_chain[key[0]] = (key, indexes)

    next_serial = 1
    for index in atom_indexes:
        try:
            next_serial = max(next_serial, int(lines[index][6:11]) + 1)
        except ValueError:
            continue

    inserts: List[Tuple[int, str]] = []
    for chain, (_key, indexes) in last_by_chain.items():
        coords = {lines[index][12:16].strip(): _pdb_xyz(lines[index]) for index in indexes}
        if "OXT" in coords:
            continue
        missing = [name for name in ("CA", "C", "O") if name not in coords]
        if missing:
            residue = lines[indexes[-1]][17:27].strip()
            raise RuntimeError(
                f"Chain {chain.strip() or '-'} residue {residue} is missing {', '.join(missing)}; cannot add OXT"
            )
        inserts.append((
            indexes[-1],
            _format_oxt(lines[indexes[-1]], next_serial, _oxt_coordinate(coords["CA"], coords["C"], coords["O"])),
        ))
        next_serial += 1

    for after, line in sorted(inserts, reverse=True):
        lines.insert(after + 1, line)
    return "\n".join(lines) + ("\n" if pdb_text.endswith("\n") or lines else "")


def _ensure_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        return mol
    return Chem.AddHs(mol, addCoords=True)


def _require_same_chemistry(poses: Sequence[Chem.Mol]) -> None:
    first = poses[0].GetNumAtoms()
    if any(pose.GetNumAtoms() != first or pose.GetNumConformers() == 0 for pose in poses):
        raise RuntimeError("MM/GBSA poses must share one hydrogenated chemistry and a conformer")


def _get_receptor(path: str) -> PreparedReceptor:
    resolved = str(Path(path).resolve())
    stat = os.stat(resolved)
    key = (resolved, stat.st_mtime_ns, stat.st_size)
    cached = _RECEPTOR_CACHE.get(key)
    if cached is None:
        if len(_RECEPTOR_CACHE) >= 4:
            _RECEPTOR_CACHE.clear()
        cached = _prepare_receptor(resolved)
        _RECEPTOR_CACHE[key] = cached
    return cached


def _prepare_receptor(path: str) -> PreparedReceptor:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    from prepare_receptors import convert_ptr

    text = _add_c_terminal_oxt(convert_ptr(Path(path).read_text(), "tyr"))
    pdb = app.PDBFile(io.StringIO(text))
    forcefield = app.ForceField(*PROTEIN_FF)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield, pH=7.4)
    system = _build_system(forcefield, modeller.topology)
    context = _context(system, modeller.topology)
    context.setPositions(modeller.positions)
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=MAX_H_STEPS)
    state = context.getState(getEnergy=True, getPositions=True)
    energy = float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
    if not math.isfinite(energy):
        raise RuntimeError(f"Receptor MM/GBSA energy is not finite for {path}")
    return PreparedReceptor(modeller.topology, state.getPositions(), energy)


def _off_molecule(mol: Chem.Mol):
    from openff.toolkit import Molecule

    molecule = Molecule.from_rdkit(mol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
    _assign_nagl_charges(molecule)
    return molecule


def _assign_nagl_charges(molecule) -> str:
    # openff-toolkit 0.16 does not register NAGL, so the .pt methods are invisible
    # on the default registry even when openff-nagl-models is installed.
    from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper

    nagl = NAGLToolkitWrapper()
    errors = []
    for model in NAGL_MODELS:
        try:
            molecule.assign_partial_charges(partial_charge_method=model, toolkit_registry=nagl)
            return model
        except Exception as exc:
            errors.append(f"{model}: {exc}")
    raise RuntimeError(
        "NAGL charge assignment failed. Install openff-nagl and openff-nagl-models.\n"
        + "\n".join(errors)
    )


def _sage_name() -> str:
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    installed = set(SMIRNOFFTemplateGenerator.INSTALLED_FORCEFIELDS)
    for name in SAGE_PREFERENCE:
        if name in installed:
            return name
    raise RuntimeError(
        "No OpenFF Sage force field is installed. Expected one of: " + ", ".join(SAGE_PREFERENCE)
    )


def _forcefield(ligand):
    import openmm.app as app
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    generator = SMIRNOFFTemplateGenerator(molecules=[ligand], forcefield=_sage_name())
    forcefield = app.ForceField(*PROTEIN_FF)
    forcefield.registerTemplateGenerator(generator.generator)
    return forcefield


def _pdb_atom_name(name: str) -> str:
    if len(name) > 4:
        raise ValueError(f"atom name {name!r} is longer than 4 characters")
    if len(name) == 4:
        return name
    return f" {name:<3s}"[:4]


def _ligand_pdb_block(mol: Chem.Mol) -> str:
    """PDB text whose atom order and names match the OpenFF molecule."""
    pose = Chem.Mol(mol)
    counts: Dict[str, int] = collections.defaultdict(int)
    for atom in pose.GetAtoms():
        symbol = atom.GetSymbol()
        counts[symbol] += 1
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName("LIG")
        info.SetResidueNumber(1)
        info.SetChainId("Z")
        info.SetIsHeteroAtom(True)
        info.SetName(_pdb_atom_name(f"{symbol}{counts[symbol]}"))
        atom.SetMonomerInfo(info)
    block = Chem.MolToPDBBlock(pose)
    if "CONECT" not in block:
        raise RuntimeError("ligand PDB is missing CONECT records, so OpenMM cannot perceive bonds")
    return block


def _ligand_topology(mol: Chem.Mol):
    import openmm.app as app

    pdb = app.PDBFile(io.StringIO(_ligand_pdb_block(mol)))
    if pdb.topology.getNumAtoms() != mol.GetNumAtoms():
        raise RuntimeError(
            f"OpenMM read {pdb.topology.getNumAtoms()} ligand atoms; RDKit has {mol.GetNumAtoms()}"
        )
    return pdb.topology


def _conformer_nm(mol: Chem.Mol) -> np.ndarray:
    conf = mol.GetConformer()
    xyz = np.array(
        [conf.GetAtomPosition(index) for index in range(mol.GetNumAtoms())],
        dtype=float,
    )
    return xyz * 0.1  # angstrom -> nanometer


def _quantity(xyz_nm: np.ndarray):
    """OpenMM position list. A bare numpy Quantity is not what Modeller.add expects."""
    import openmm
    import openmm.unit as unit

    vectors = [
        openmm.Vec3(float(row[0]), float(row[1]), float(row[2]))
        for row in np.asarray(xyz_nm, dtype=float)
    ]
    return vectors * unit.nanometer


def _merge(receptor: PreparedReceptor, ligand_topology, ligand_nm: np.ndarray):
    import openmm.app as app

    modeller = app.Modeller(receptor.topology, receptor.positions)
    modeller.add(ligand_topology, _quantity(ligand_nm))
    return modeller.topology, modeller.positions


def _build_system(forcefield, topology):
    import openmm.app as app

    return forcefield.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )


def _freeze_heavy(system, topology) -> None:
    import openmm.unit as unit

    for atom in topology.atoms():
        element = atom.element
        if element is None or element.symbol != "H":
            system.setParticleMass(atom.index, 0.0 * unit.dalton)


def _cpu_threads() -> str:
    raw = os.environ.get("OPENMM_CPU_THREADS", "").strip()
    if raw.isdigit() and int(raw) >= 1:
        return raw
    return "1"


def _context(system, topology, freeze_heavy: bool = True):
    """CPU context. The platform is explicit so OpenCL is never opened."""
    import openmm
    import openmm.unit as unit

    if freeze_heavy:
        _freeze_heavy(system, topology)
    platform = openmm.Platform.getPlatformByName("CPU")
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    return openmm.Context(system, integrator, platform, {"Threads": _cpu_threads()})


def _minimized_state(context, positions):
    """Relax hydrogens. Heavy atoms are frozen by a zero mass on this context."""
    import openmm
    import openmm.unit as unit

    context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=MAX_H_STEPS)
    state = context.getState(getEnergy=True, getPositions=True)
    energy = float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
    return energy, state.getPositions()


def _potential_energy(context, positions) -> float:
    import openmm.unit as unit

    context.setPositions(positions)
    state = context.getState(getEnergy=True)
    return float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))


def _join_positions(receptor_positions, ligand_positions):
    import openmm.unit as unit

    receptor_nm = list(receptor_positions.value_in_unit(unit.nanometer))
    ligand_nm = list(ligand_positions.value_in_unit(unit.nanometer))
    return unit.Quantity(receptor_nm + ligand_nm, unit.nanometer)
