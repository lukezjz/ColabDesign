"""
This script is used for testing new functions for ColabDesign
"""
import jax
import jax.numpy as jnp
from colabdesign import mk_afdesign_model, clear_mem


# Path to model parameters of AF2 in my PC
params_path = "/home/zhangjizhong/Database/Alphafold2/params"


def rewire():
    clear_mem()
    model = mk_afdesign_model(protocol="partial", data_dir=params_path, use_templates=True, use_multimer=False)  # set True to constrain positions using template input
    model.prep_inputs(pdb_filename="Inputs/2mr5.pdb", chain="A", length=100, fix_seq=True,
                      pos="4-9,13-26,30-33,37-50,54-59,63-76,80-84,112-125",
                      fix_pos="4-9,13-26,30-33,37-50,54-59,63-76,80-84,112-125")
    model.rewire(order=[1, 2, 3, 0, 7, 4, 5, 6],
                 loops=[3, 3, 3, 4, 3, 3, 4],
                 offset=0)
    model.restart()
    model.set_weights(dgram_cce=1, con=0)
    model.design_3stage(300, 100, 10)


# ColabDesign for affibody engineering
def affibody():
    # Configurations about the affibody
    affibody_path = "Inputs/1q2n.pdb"
    affibody_pos = "9-11,13-14,17-18,24-25,27-28,32,35"    # select resi 9-11+13-14+17-18+24-25+27-28+32+35; show stick, sele

    # Configurations about the target protein
    pdb_filename = "Inputs/1h0t_A.pdb"
    output_pdb = "Outputs/1h0t_affibody.pdb"

    clear_mem()
    model = mk_afdesign_model(protocol="affibody", data_dir=params_path, use_templates=True, use_multimer=True,
                                  affibody_path=affibody_path, affibody_pos=affibody_pos, num_recycles=1, recycle_mode="last")
    model.prep_inputs(pdb_filename=pdb_filename, chain="A")
    model.restart(mode=["soft", "gumbel", "wildtype"])
    model.design_3stage(100, 100, 10)
    model.save_pdb(output_pdb)

    lines = []
    with open(output_pdb) as fr:
        for line in fr:
            if line == "ENDMDL\n":
                break
            lines.append(line)
    with open(output_pdb, "w") as fw:
        fw.writelines(lines)


# ColabDesign for BRIL fusion for structure determinant using Cryo-EM
def BRIL():
    # Configuration about BRIL
    BRIL_path = "Inputs/6cbv_A.pdb"
    BRIL_frags = "3-100"

    # Configurations about the target protein
    target_path = "Inputs/1h0t_A.pdb"
    target_frags = "1-18,24-58"

    # Configurations about linker lengths
    linker_lengths = "5,6"

    output_pdb = "Outputs/1h0t_BRIL.pdb"

    clear_mem()
    model = mk_afdesign_model(protocol="BRIL", data_dir=params_path, use_templates=True, use_multimer=False, # num_recycles=3, recycle_mode="average",
                              BRIL_path=BRIL_path, BRIL_frags=BRIL_frags)
    model.prep_inputs(pdb_filename=target_path, chain="A", frags=target_frags, linker_lengths=linker_lengths)
    model.restart(mode=["soft", "gumbel", "wildtype"])
    model.design_3stage(100,100, 10)
    model.save_pdb(output_pdb)

    lines = []
    with open(output_pdb) as fr:
        for line in fr:
            if line == "ENDMDL\n":
                break
            lines.append(line)
    with open(output_pdb, "w") as fw:
        fw.writelines(lines)

BRIL()
