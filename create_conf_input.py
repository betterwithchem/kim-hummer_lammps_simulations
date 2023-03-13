#!/bin/env python3

import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=" ")

parser.add_argument("-f", dest="pdb_filename", required=True, type=str, 
                    help="PDB file of the starting configuration.")
parser.add_argument("-nmol", dest="nmolecules", required=True, type=int, nargs='+', 
                    help="number of molecules of each species in the system.")
parser.add_argument("-lmol", dest="length_molecules", required=True, nargs='+', type=int, 
                    help="number of atoms in each molecule type. Same number of arguments and same order as -nmol.")
parser.add_argument("-o", dest="output_conf_file", required=True, type=str, 
                    help="name of the output file with the starting conformation and topology")

parser.add_argument("-rigid", dest="rigid_ranges", type=int, nargs='+', 
                    help="list of ranges of atoms that belong to rigid bodies. \
                    Only an even number of arguments is accepted. No arguments == no rigid bodies. Ranges include the termini.")
parser.add_argument("-rigidfile", dest="rigid_file", type=str, 
                    help="name of the file used for fix rigid commands")

args	=	parser.parse_args()

################# SANITY CHECKS OF THE INPUT ARGUMENTS ###############

if len(args.length_molecules)!=len(args.nmolecules):
    print("Error! the number of molecules in the system with -nmol (%d) is different from the number of 
          lengths given with -lmol (%d)" % (len(args.length_molecules), len(args.nmolecules)))
    sys.exit(1)

rigid_atoms=[]
if args.rigid_ranges is not None:
    if (len(args.rigid_ranges)%2)!=0:
        print("Error! the number of arguments given with -rigid is not an even number.")
        sys.exit(1)
    elif args.rigid_file is None:
        print("Error! you have not provided a name for the rigid data file")
        sys.exit(1)
        
#######################################################################
        
if args.rigid_ranges is not None: 
    for iel in range(0,len(args.rigid_ranges),2):
        start=args.rigid_ranges[iel]
        end=args.rigid_ranges[iel+1]
        for i in range(start,end+1):
            rigid_atoms.append(i)

    rigid_f=open(args.rigid_file,'w')
        
    for iel in range(0,len(args.rigid_ranges),2):
        rigid_f.write("group\tgroup_%d id %d:%d\n" % (int(iel/2), args.rigid_ranges[iel], args.rigid_ranges[iel+1]) )
    # ^: define the groups of rigid atoms. One group for each molecule.

    rigid_f.write("group\tglob union ")
    for iel in range(0,len(args.rigid_ranges),2):
        rigid_f.write("group_%d " %  (int(iel/2)))
    rigid_f.write("\n\n")
    # ^: union of the rigid-atom groups for a global group (called "glob", as in globular)

    rigid_f.write("fix\trigidFix all rigid group %d " % (int(len(args.rigid_ranges)/2)) )
    for i in range(0,(int(len(args.rigid_ranges)/2))):
        rigid_f.write("group_%d " % (i))
    rigid_f.write("\n\n")
    # ^: apply fix rigid to rigid atoms

    for i in range(0,(int(len(args.rigid_ranges)/2))):
        rigid_f.write("neigh_modify\texclude group group_%d group_%d\n" % (i,i))
    # ^: exclude intramolecular non-bonded interactions for rigid atoms
    #    bonds between rigid atoms are excluded in the general LAMMPS input file
    
### Parameters of the atom/residue types ###

aatype_dict={"ALA":1,"CYS":2,"ASP":3,"GLU":4,"PHE":5,"GLY":6,"HIS":7,"ILE":8,"LYS":9,"LEU":10,"MET":11,
             "ASN":12,"PRO":13,"GLN":14,"ARG":15,"SER":16,"THR":17,"VAL":18,"TRP":19,"TYR":20}

masses={"GLY":57.05,"ALA":71.08,"CYS":103.1,"ASP":115.1,"GLU":129.1,"PHE":147.2,"HIS":137.1,"ILE":113.2, 
        "LYS":128.2,"LEU":113.2,"MET":131.2,"ASN":114.1,"PRO":97.12,"GLN":128.1,"ARG":156.2,"SER":87.08,
        "THR":101.1,"VAL":99.07,"TRP":186.2,"TYR":163.2}

charges={"GLY":0,"ALA":0,"CYS":0,"ASP":-1,"GLU":-1,"PHE":0,"HIS":0.5,"ILE":0,"LYS":1,"LEU":0,"MET":0,"ASN":0,
         "PRO":0,"GLN":0,"ARG":1,"SER":0,"THR":0,"VAL":0,"TRP":0,"TYR":0}

############################################

atomid=[]
atomtype=[]
sequence=[]

x=[]
y=[]
z=[]

got_box_sizes=False

with open(args.pdb_filename,'r') as f:
    i=0 # counter for the atom_id
    for line in f:
        if line[0:4]=="ATOM":
            i+=1
            atomid.append(i)
            # ^: atomids are consecutive numbers starting from 1, numbers in the original PDB file are NOT considered.
            isrigid=(i in rigid_atoms)
            # ^: flag the atom if it is rigid
            aa=line[17:20]
            # ^: residue type as in the original PDB (3 letter code)
            atomtype.append(aatype_dict[aa]+isrigid*20)
            # ^: save the atomtype discriminating between disordered and globular types
            sequence.append(aa)
            # ^: save the sequence

            x.append(float(line[30:38]))
            y.append(float(line[38:46]))
            z.append(float(line[46:54]))
            # ^: get the coordinates (in A!!!)
            
        elif line[0:6]=="CRYST1":
            col=line.split()
            box_sizes=[float(col[1]), float(col[2]), float(col[3])]
            # ^: get the box size (in A!!!)
            got_box_sizes=True

if not got_box_sizes:
    print("Error! the box dimensions have not been found in the PDB file")
    sys.exit(1)
            
natoms=len(atomid)

theoret_natoms=0


for i in range(len(args.length_molecules)):
    theoret_natoms+=(args.nmolecules[i]*args.length_molecules[i])

if natoms!=theoret_natoms:
    print("Error! PDB file does NOT contain the expected number of atoms (i.e. sum(length_molecules))")
    print("expected atoms = %d, found atoms = %d" % (theoret_natoms, natoms))
    sys.exit(1)


nmolecules=np.sum(args.nmolecules)
    
chain_ranges=[[0, 0] for i in range(nmolecules)]

prev_end=0
prev_species_ichain=0
for ispecies in range(len(args.nmolecules)):
    for ichain in range(prev_species_ichain,args.nmolecules[ispecies]+prev_species_ichain):
        chain_ranges[ichain][0]=prev_end +1
        chain_ranges[ichain][1]=chain_ranges[ichain][0] + args.length_molecules[ispecies] -1
        prev_end = chain_ranges[ichain][1]
    prev_species_ichain=ichain+1

chain=[]
for iatom in range(1,natoms+1):
    for ichain in range(len(chain_ranges)):
        if iatom in range(chain_ranges[ichain][0], chain_ranges[ichain][1]+1):
            chain.append(ichain+1)

print(chain_ranges)

bonds=len(atomid)-nmolecules
natomtypes=40
nbondtypes=1
            
conf_f=open(args.output_conf_file,'w')

conf_f.write("Initial Comment Line\n\n")

conf_f.write("\t%d\tatoms\n" % (natoms))
conf_f.write("\t%d\tbonds\n" % (bonds))

conf_f.write("\n")

conf_f.write("\t%d\tatom types\n" % (natomtypes))
conf_f.write("\t%d\tbond types\n" % (nbondtypes))

conf_f.write("\n")

conf_f.write("\t%f %f\txlo xhi\n" % (-box_sizes[0]/2, box_sizes[0]/2))
conf_f.write("\t%f %f\tylo yhi\n" % (-box_sizes[1]/2, box_sizes[1]/2))
conf_f.write("\t%f %f\tzlo zhi\n" % (-box_sizes[2]/2, box_sizes[2]/2))

conf_f.write("\n")

conf_f.write("Masses\n\n")

for aatype in aatype_dict:	# type 1 to 20, disordered
    conf_f.write("\t%d\t%f\n" % (aatype_dict[aatype], masses[aatype]))
for aatype in aatype_dict:	# type 21 to 40, ordered
    conf_f.write("\t%d\t%f\n" % (aatype_dict[aatype]+20, masses[aatype]))

conf_f.write("\n")

conf_f.write("Atoms\n\n")

for iatom in range(natoms):
    conf_f.write("\t%d\t%d\t%d\t%f\t%f\t%f\t%f\n" % (iatom+1, chain[iatom], atomtype[iatom], charges[sequence[iatom]], x[iatom], y[iatom], z[iatom]))

conf_f.write("\n\n")

conf_f.write("Bonds\n\n")

ibond=0
for ichain in range(nmolecules):
    for iatom in range(chain_ranges[ichain][0], chain_ranges[ichain][1]):
        conf_f.write("\t%d\t%d\t%d\t%d\n" % (ibond+1, 1, iatom, iatom+1))
        ibond+=1

conf_f.write("\n\nBond Coeffs\n\n")

conf_f.write("\t1\t%f\t3.8\n" % (10/4.184))

