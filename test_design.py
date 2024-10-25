"""
This script is used for testing new functions for ColabDesign
"""
import jax
import jax.numpy as jnp
from colabdesign import mk_afdesign_model, clear_mem


# Path to model parameters of AF2 in my PC
params_path = "/home/zhangjizhong/Database/Alphafold2/params"


# ColabDesign for affibody engineering
def affibody():
    # Configurations about affibody
    affibody_path = "Inputs/1q2n.pdb"
    affibody_pos = "9-11,13-14,17-18,24-25,27-28,32,35"    # select resi 9-11+13-14+17-18+24-25+27-28+32+35; show stick, sele
    pdb_filename = "Inputs/1h0t_A.pdb"
    output_pdb = "Outputs/1h0t_affibody.pdb"

    clear_mem()
    model = mk_afdesign_model(protocol="affibody", data_dir=params_path, use_templates=True, use_multimer=True,
                              affibody_path=affibody_path, affibody_pos=affibody_pos, num_recycles=3, recycle_mode="average")
    model.prep_inputs(pdb_filename=pdb_filename, chain="A")
    model.restart(mode=["soft", "gumbel", "wildtype"])
    model.design_3stage(100, 100, 10)
    model.save_pdb(f"Outputs/1h0t_affibody.pdb")

    lines = []
    with open(output_pdb) as fr:
        for line in fr:
            if line == "ENDMDL\n":
                break
            lines.append(line)
    with open(output_pdb, "w") as fw:
        fw.writelines(lines)

affibody()
