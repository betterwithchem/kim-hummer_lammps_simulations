#!/MASTER/PROG/Python-3.6/bin/python3

#modded by MP on 12th November 2018 to add parser and make it "loopable"
#modified by KP on 17th November 2021 to write the KH parameters in LAMMPS format
#last update 17/11/2021

#Example: python3 file_non_bonded_KimHummer.py -o KH_pair_coeffs

import numpy as np
import math
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=" ")

parser.add_argument("-o",dest="outputFile",required=True,type=str)

args	=	parser.parse_args()

rlimit=0
delr=0.002 #nm
rcut=10.0 #nm
D=80 #permitivity of water
lam=1.0 #nm #lambda Debye-Huckel for 100 mM as proposed in the Kim&Hummer paper (1nm)
kbT=0.008314*0.239006*300 #kcal.mol-1 (LAMMPS units)

#Two rescaling params for espij=ulambda*(eij-e0) (see Kim&Hummer paper)

ulambda_D=0.228 # [-] model D 
e0_D=-1.67 # model D (multiplied later by kbT)      #kbT units  #offset == -1 kcal/mol

ulambda_A=0.159 # [-] model A 
e0_A=-2.27 # model A (multiplied later by kbT)  #kbT units  #offset == -1.35 kcal/mol

#Data
oneLetterCode={"ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y"}

aaTable=["ALA","CYS","ASP","GLU","PHE","GLY","HIS","ILE","LYS","LEU","MET","ASN","PRO","GLN","ARG","SER","THR","VAL","TRP","TYR"]

bb_residues={"GLY":"BBG", "ALA":"BBA", "CYS":"BBC", "ASP":"BBD", "GLU":"BBE", "PHE":"BBF", "HIS":"BBH", "ILE":"BBI", "LYS":"BBK", "LEU":"BBL", "MET":"BBM", "ASN":"BBN", "PRO":"BBP", "GLN":"BBQ", "ARG":"BBR", "SER":"BBS", "THR":"BBT", "VAL":"BBV", "TRP":"BBW", "TYR":"BBY"}

sigmas	=  {"BBG":4.50, "BBA": 5.04, "BBC":5.48, "BBD":5.58 , "BBE":5.92, "BBF":6.36, "BBH":6.08, "BBI":6.18, "BBK":6.36, "BBL":6.18, "BBM":6.18, "BBN":5.68, "BBP":5.56, "BBQ":6.02, "BBR":6.56, "BBS":5.18, "BBT":5.62, "BBV":5.86, "BBW":6.78, "BBY":6.46}

oneNumberCode_D={"ALA":"1","CYS":"2","ASP":"3","GLU":"4","PHE":"5","GLY":"6","HIS":"7","ILE":"8","LYS":"9","LEU":"10","MET":"11","ASN":"12","PRO":"13","GLN":"14","ARG":"15","SER":"16","THR":"17","VAL":"18","TRP":"19","TYR":"20"}

oneNumberCode_A={"ALA":"21","CYS":"22","ASP":"23","GLU":"24","PHE":"25","GLY":"26","HIS":"27","ILE":"28","LYS":"29","LEU":"30","MET":"31","ASN":"32","PRO":"33","GLN":"34","ARG":"35","SER":"36","THR":"37","VAL":"38","TRP":"39","TYR":"40"}


##### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ####
# IMPORTANT!!! Since for model D S-S interaction would be 0  #
# (not even repulsive C12), SERSER coefficient is changed,   #
# such that the final eij is 0.001, 			             #
# The same applies for GLUARG and CYSGLU in model A		     #
##### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ####

hydrophobicity={'ALAALA':-2.72, 'ALACYS':-3.57, 'ALAASP':-1.70, 'ALAGLU':-1.51, 'ALAPHE':-4.81, 'ALAGLY':-2.31, 'ALAHIS':-2.41, 'ALAILE':-4.58, 'ALALYS':-1.31, 'ALALEU':-4.91, 'ALAMET':-3.94, 'ALAASN':-1.84, 'ALAPRO':-2.03, 'ALAGLN':-1.89, 'ALAARG':-1.83, 'ALASER':-2.01, 'ALATHR':-2.32, 'ALAVAL':-4.04, 'ALATRP':-3.82, 'ALATYR':-3.36, 'CYSCYS':-5.44, 'CYSASP':-2.41, 'CYSGLU':-2.269, 'CYSPHE':-5.80, 'CYSGLY':-3.16, 'CYSHIS':-3.60, 'CYSILE':-5.50, 'CYSLYS':-1.95, 'CYSLEU':-5.83, 'CYSMET':-4.99, 'CYSASN':-2.59, 'CYSPRO':-3.07, 'CYSGLN':-2.85, 'CYSARG':-2.57, 'CYSSER':-2.86, 'CYSTHR':-3.11, 'CYSVAL':-4.96, 'CYSTRP':-4.95, 'CYSTYR':-4.16, 'ASPASP':-1.21, 'ASPGLU':-1.02, 'ASPPHE':-3.48, 'ASPGLY':-1.59, 'ASPHIS':-2.32, 'ASPILE':-3.17, 'ASPLYS':-1.68, 'ASPLEU':-3.40, 'ASPMET':-2.57, 'ASPASN':-1.68, 'ASPPRO':-1.33, 'ASPGLN':-1.46, 'ASPARG':-2.29, 'ASPSER':-1.63, 'ASPTHR':-1.80, 'ASPVAL':-2.48, 'ASPTRP':-2.84, 'ASPTYR':-2.76, 'GLUGLU':-0.91, 'GLUPHE':-3.56, 'GLUGLY':-1.22, 'GLUHIS':-2.15, 'GLUILE':-3.27, 'GLULYS':-1.80, 'GLULEU':-3.59, 'GLUMET':-2.89, 'GLUASN':-1.51, 'GLUPRO':-1.26, 'GLUGLN':-1.42, 'GLUARG':-2.269, 'GLUSER':-1.48, 'GLUTHR':-1.74, 'GLUVAL':-2.67, 'GLUTRP':-2.99, 'GLUTYR':-2.79, 'PHEPHE':-7.26, 'PHEGLY':-4.13, 'PHEHIS':-4.77, 'PHEILE':-6.84, 'PHELYS':-3.36, 'PHELEU':-7.28, 'PHEMET':-6.56, 'PHEASN':-3.75, 'PHEPRO':-4.25, 'PHEGLN':-4.10, 'PHEARG':-3.98, 'PHESER':-4.02, 'PHETHR':-4.28, 'PHEVAL':-6.29, 'PHETRP':-6.16, 'PHETYR':-5.66, 'GLYGLY':-2.24, 'GLYHIS':-2.15, 'GLYILE':-3.78, 'GLYLYS':-1.15, 'GLYLEU':-4.16, 'GLYMET':-3.39, 'GLYASN':-1.74, 'GLYPRO':-1.87, 'GLYGLN':-1.66, 'GLYARG':-1.72, 'GLYSER':-1.82, 'GLYTHR':-2.08, 'GLYVAL':-3.38, 'GLYTRP':-3.42, 'GLYTYR':-3.01, 'HISHIS':-3.05, 'HISILE':-4.14, 'HISLYS':-1.35, 'HISLEU':-4.54, 'HISMET':-3.98, 'HISASN':-2.08, 'HISPRO':-2.25, 'HISGLN':-1.98, 'HISARG':-2.16, 'HISSER':-2.11, 'HISTHR':-2.42, 'HISVAL':-3.58, 'HISTRP':-3.98, 'HISTYR':-3.52, 'ILEILE':-6.54, 'ILELYS':-3.01, 'ILELEU':-7.04, 'ILEMET':-6.02, 'ILEASN':-3.24, 'ILEPRO':-3.76, 'ILEGLN':-3.67, 'ILEARG':-3.63, 'ILESER':-3.52, 'ILETHR':-4.03, 'ILEVAL':-6.05, 'ILETRP':-5.78, 'ILETYR':-5.25, 'LYSLYS':-0.12, 'LYSLEU':-3.37, 'LYSMET':-2.48, 'LYSASN':-1.21, 'LYSPRO':-0.97, 'LYSGLN':-1.29, 'LYSARG':-0.59, 'LYSSER':-1.05, 'LYSTHR':-1.31, 'LYSVAL':-2.49, 'LYSTRP':-2.69, 'LYSTYR':-2.60, 'LEULEU':-7.37, 'LEUMET':-6.41, 'LEUASN':-3.74, 'LEUPRO':-4.20, 'LEUGLN':-4.04, 'LEUARG':-4.03, 'LEUSER':-3.92, 'LEUTHR':-4.34, 'LEUVAL':-6.48, 'LEUTRP':-6.14, 'LEUTYR':-5.67, 'METMET':-5.46, 'METASN':-2.95, 'METPRO':-3.45, 'METGLN':-3.30, 'METARG':-3.12, 'METSER':-3.03, 'METTHR':-3.51, 'METVAL':-5.32, 'METTRP':-5.55, 'METTYR':-4.91, 'ASNASN':-1.68, 'ASNPRO':-1.53, 'ASNGLN':-1.71, 'ASNARG':-1.64, 'ASNSER':-1.58, 'ASNTHR':-1.88, 'ASNVAL':-2.83, 'ASNTRP':-3.07, 'ASNTYR':-2.76, 'PROPRO':-1.75, 'PROGLN':-1.73, 'PROARG':-1.70, 'PROSER':-1.57, 'PROTHR':-1.90, 'PROVAL':-3.32, 'PROTRP':-3.73, 'PROTYR':-3.19, 'GLNGLN':-1.54, 'GLNARG':-1.80, 'GLNSER':-1.49, 'GLNTHR':-1.90, 'GLNVAL':-3.07, 'GLNTRP':-3.11, 'GLNTYR':-2.97, 'ARGARG':-1.55, 'ARGSER':-1.62, 'ARGTHR':-1.90, 'ARGVAL':-3.07, 'ARGTRP':-3.41, 'ARGTYR':-3.16, 'SERSER':-1.669, 'SERTHR':-1.96, 'SERVAL':-3.05, 'SERTRP':-2.99, 'SERTYR':-2.78, 'THRTHR':-2.12, 'THRVAL':-3.46, 'THRTRP':-3.22, 'THRTYR':-3.01, 'VALVAL':-5.52, 'VALTRP':-5.18, 'VALTYR':-4.62, 'TRPTRP':-5.06, 'TRPTYR':-4.66, 'TYRTYR':-4.17,}

#Tables
myTableFile=open(args.outputFile+".data",'w') #MP
for i in range(len(aaTable)):
    
    for j in range(i,len(aaTable)):
        
        sigma=(sigmas[bb_residues[aaTable[i]]] + sigmas[bb_residues[aaTable[j]]]) / 2
        rlimit=(2**(1/6)) * sigma
        
        eij_D=hydrophobicity[aaTable[i]+aaTable[j]]
        epsij_D=ulambda_D*(eij_D-e0_D)*kbT    #units purpose from kbT to kcal/mol
        myTableFile.write("{:s}\t{:s}\t{:s}\t{:s}\t{:6.8f}\t{:6.2f}\n".format("pair_coeff",oneNumberCode_D[aaTable[i]],oneNumberCode_D[aaTable[j]],"lj/humm",epsij_D,sigma))

for i in range(len(aaTable)):
       
    for j in range(i,len(aaTable)):
        
        sigma=(sigmas[bb_residues[aaTable[i]]] + sigmas[bb_residues[aaTable[j]]]) / 2
        rlimit=(2**(1/6)) * sigma
        
        eij_A=hydrophobicity[aaTable[i]+aaTable[j]]
        epsij_A=ulambda_A*(eij_A-e0_A)*kbT    
        myTableFile.write("{:s}\t{:s}\t{:s}\t{:s}\t{:6.8f}\t{:6.2f}\n".format("pair_coeff",oneNumberCode_A[aaTable[i]],oneNumberCode_A[aaTable[j]],"lj/humm",epsij_A,sigma))

for i in range(len(aaTable)):
       
    for j in range(0,len(aaTable)):
        
        sigma=(sigmas[bb_residues[aaTable[i]]] + sigmas[bb_residues[aaTable[j]]]) / 2
        rlimit=(2**(1/6)) * sigma
        
        if i<=j:  
            eij_A=hydrophobicity[aaTable[i]+aaTable[j]]
            epsij_A=ulambda_A*(eij_A-e0_A)*kbT  
            myTableFile.write("{:s}\t{:s}\t{:s}\t{:s}\t{:6.8f}\t{:6.2f}\n".format("pair_coeff",oneNumberCode_D[aaTable[i]],oneNumberCode_A[aaTable[j]],"lj/humm",epsij_A,sigma))
        else:
            eij_A=hydrophobicity[aaTable[j]+aaTable[i]]            
            epsij_A=ulambda_A*(eij_A-e0_A)*kbT   
            myTableFile.write("{:s}\t{:s}\t{:s}\t{:s}\t{:6.8f}\t{:6.2f}\n".format("pair_coeff",oneNumberCode_D[aaTable[i]],oneNumberCode_A[aaTable[j]],"lj/humm",epsij_A,sigma))
        
myTableFile.close()
