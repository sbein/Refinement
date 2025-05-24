import os
import sys
from glob import glob

import uproot
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, DelphesSchema


# use LCG 104 cuda

isTest = False
nTest = 200  # 2

# careful, the fullsimpath is actually not used!

# fastsimpath = '/data/dust/user/beinsam/FastSim/CMSSW_10_6_22/src/TTbar/Fast/*/step3_inNANOAODSIM.root'
# fullsimpath = '/data/dust/user/beinsam/FastSim/CMSSW_10_6_22/src/TTbar/Full/*/step3_inNANOAODSIM.root'
# fastsimpath = '/data/dust/user/beinsam/FastSim/CMSSW_10_6_22/src/TTbar/Fast/1105/step3_inNANOAODSIM.root'
# fullsimpath = '/data/dust/user/beinsam/FastSim/CMSSW_10_6_22/src/TTbar/Full/1105/step3_inNANOAODSIM.root'
# fastsimpath = '/data/dust/user/beinsam/FastSim/CMSSW_10_6_22/src/Dijet600_800/Fast/*/*inNANOAODSIM.root'
# fullsimpath = '/data/dust/user/beinsam/FastSim/CMSSW_10_6_22/src/Dijet600_800/Full/*/*inNANOAODSIM.root'

# fastsimpath = '/data/dust/user/beinsam/FastSim/VisitCaloResponse/CMSSW_12_2_3/src/T1tttt/Fast/*/*NANO*.root'
# fastsimpath = '/data/dust/user/beinsam/FastSim/VisitCaloResponse/CMSSW_12_2_3/src/T1tttt/FastGenDecays/*/step3_inNANOAODSIM.root'
# fullsimpath = '/data/dust/user/beinsam/FastSim/VisitCaloResponse/CMSSW_12_2_3/src/T1tttt/Full/*/step3_inNANOAODSIM.root'

# fastsimpath = '/data/dust/user/beinsam/FastSim/VisitCaloResponse/CMSSW_12_2_3/src/TTbar/FastGenDecays/*/step3_inNANOAODSIM.root'
# fullsimpath = '/data/dust/user/beinsam/FastSim/VisitCaloResponse/CMSSW_12_2_3/src/TTbar/Full/*/step3_inNANOAODSIM.root'

# fastsimpath = '/data/dust/user/beinsam/FastSim/VisitCaloResponse/CMSSW_12_2_3/src/TTbar/Fast/*/*NANO*.root'
# fullsimpath = '/data/dust/user/beinsam/FastSim/VisitCaloResponse/CMSSW_12_2_3/src/TTbar/Fast/*/*NANO*.root'

# run 3
# fastsimpath = '/data/dust/user/beinsam/FastSim/Refinement/CMSSW_12_6_0/src/TTbar/Fast/*/step2_SIM_RECOBEFMIX_DIGI_L1_DIGI2RAW_L1Reco_RECO_PAT_NANO.root'
# fullsimpath = '/data/dust/user/beinsam/FastSim/Refinement/CMSSW_12_4_11_patch3/src/TTbar/Full/*/step3_NANO.root'
# fullsimpath = '/data/dust/user/beinsam/FastSim/Refinement/CMSSW_12_6_0/src/TTbar/Full/*/*NANO.root'

# fastsimpath = '/data/dust/user/beinsam/FastSim/Refinement/Backport2Decayer/CMSSW_10_6_X_2023-04-23-0000/src/TTbar/Fast/*/TTbar_13TeV_TuneCUETP8M1_cfi_GEN_SIM_RECOBEFMIX_DIGI_L1_DIGI2RAW_L1Reco_RECO_PAT_NANO.root'
# fullsimmodifier = lambda x: x.replace('/Fast/', '/Full/').replace('TTbar_13TeV_TuneCUETP8M1_cfi_GEN_SIM_RECOBEFMIX_DIGI_L1_DIGI2RAW_L1Reco_RECO_PAT_NANO.root', 'step3_inNANOAODSIM.root')

# fastsimpath = '/data/dust/user/beinsam/FastSim/Refinement/Backport2Decayer/CMSSW_10_6_X_2023-04-23-0000/src/T1tttt/FastNoRefine/*/SMS-T1tttt_mGl-1500_mLSP-100_13TeV-pythia8_cfi_GEN_SIM_RECOBEFMIX_DIGI_L1_DIGI2RAW_L1Reco_RECO_PAT_NANO.root'
# fullsimmodifier = lambda x: x.replace('/FastNoRefine/', '/Full/').replace('SMS-T1tttt_mGl-1500_mLSP-100_13TeV-pythia8_cfi_GEN_SIM_RECOBEFMIX_DIGI_L1_DIGI2RAW_L1Reco_RECO_PAT_NANO.root', 'step3_inNANOAODSIM.root')

fastsimpath = '/data/dust/user/beinsam/FastSim/Refinement/TrainingSamples/CMSSW_13_0_13/src/TTbarRun3/Fast/*/*NANO.root'
fullsimmodifier = lambda x: x.replace('/Fast/', '/Full/').replace('step2_SIM_RECOBEFMIX_DIGI_L1_DIGI2RAW_L1Reco_RECO_PAT_NANO.root', 'step2_PAT_NANO.root')

fastsimpath = '/data/dust/user/beinsam/FastSim/Refinement/TrainingSamples/CMSSW_13_0_13/src/T1ttttRun3/Fast/*/*NANO.root'
fullsimmodifier = lambda x: x.replace('/Fast/', '/Full/').replace('step2_SIM_RECOBEFMIX_DIGI_L1_DIGI2RAW_L1Reco_RECO_PAT_NANO.root', 'step2_PAT_NANO.root')

fastsimpath = '/data/dust/user/beinsam/FastSim/Refinement/TrainingSamples/CMSSW_14_0_12/src/T1ttttRun3PU/Fast/*/step2_SIM_RECOBEFMIX_DIGI_L1_DIGI2RAW_L1Reco_RECO_PAT_NANO_PU.root'
fullsimmodifier = lambda x: x.replace('/Fast/', '/Full/').replace('step2_SIM_RECOBEFMIX_DIGI_L1_DIGI2RAW_L1Reco_RECO_PAT_NANO_PU.root', 'NANOAODSIM.root')

# if isTest:
#     fastsimpath = fastsimpath.replace('/*/', '/1/')
#     fullsimpath = fullsimpath.replace('/*/', '/1/')

fastsimlist = glob(fastsimpath)
fastsimlist = sorted(fastsimlist)

# fullsimlist = glob(fullsimpath)
# fullsimlist = sorted(fullsimlist)

if isTest: fastsimlist = fastsimlist[:nTest]


geneventvars = ['nGenJet', 'nGenJetAK8', 'GenMET_phi', 'GenMET_pt', 'FiducialMET_phi', 'FiducialMET_pt']
addgeneventvars = [
    
]

receventvars = ['nJet', 'nFatJet', 'PV_npvs', 'PV_npvsGood',
                # 'CaloMET_phi', 'CaloMET_pt', 'CaloMET_sumEt',
                'PFMET_phi', 'PFMET_pt', 'PFMET_sumEt',
                'PuppiMET_phi', 'PuppiMET_pt', 'PuppiMET_sumEt',
                'RawPFMET_phi', 'RawPFMET_pt', 'RawPFMET_sumEt',
                'RawPuppiMET_phi', 'RawPuppiMET_pt', 'RawPuppiMET_sumEt',
                # 'TrkMET_phi', 'TrkMET_pt', 'TrkMET_sumEt'
                ]
addreceventvars = [
    # 'HT', 'MHT_phi', 'MHT_pt'  # TODO: uncomment (M)HT?
]

genjetvars = ['eta', 'mass', 'phi', 'pt', 'hadronFlavour', 'nBHadrons', 'nCHadrons', 'partonFlavour']
addgenjetvars = [
    'nearest_dR', 'nearest_pt'
]

# recjetvars = ['area', 'bRegCorr', 'bRegRes', 'btagCSVV2', 'btagDeepB', 'btagDeepCvB', 'btagDeepCvL', 'btagDeepFlavB', 'btagDeepFlavCvB', 'btagDeepFlavCvL', 'btagDeepFlavQG', 'cRegCorr', 'cRegRes', 'chEmEF', 'chHEF', 'cleanmask', 'electronIdx1', 'electronIdx2', 'eta', 'genJetIdx', 'hadronFlavour', 'jetId', 'mass', 'muEF', 'muonIdx1', 'muonIdx2', 'muonSubtrFactor', 'nConstituents', 'nElectrons', 'nMuons', 'neEmEF', 'neHEF', 'partonFlavour', 'phi', 'pt', 'puId', 'puIdDisc', 'qgl', 'rawFactor']
# recjetvars = ['area', 'bRegCorr', 'bRegRes', 'btagCSVV2', 'btagDeepB', 'btagDeepCvB', 'btagDeepCvL', 'btagDeepFlavB', 'btagDeepFlavCvB', 'btagDeepFlavCvL', 'btagDeepFlavQG', 'cRegCorr', 'cRegRes', 'chEmEF', 'chFPV0EF', 'chHEF', 'cleanmask', 'electronIdx1', 'electronIdx2', 'eta', 'genJetIdx', 'hadronFlavour', 'hfadjacentEtaStripsSize', 'hfcentralEtaStripSize', 'hfsigmaEtaEta', 'hfsigmaPhiPhi', 'jetId', 'mass', 'muEF', 'muonIdx1', 'muonIdx2', 'muonSubtrFactor', 'nConstituents', 'nElectrons', 'nMuons', 'neEmEF', 'neHEF', 'partonFlavour', 'phi', 'pt', 'puId', 'puIdDisc', 'qgl', 'rawFactor']
# recjetvars = ['jetId', 'nConstituents', 'nElectrons', 'nMuons', 'nSVs', 'area', 'btagDeepFlavB', 'btagDeepFlavCvB', 'btagDeepFlavCvL', 'btagDeepFlavQG', 'btagRobustParTAK4B', 'btagRobustParTAK4CvB', 'btagRobustParTAK4CvL', 'btagRobustParTAK4QG', 'chEmEF', 'chHEF', 'eta', 'mass', 'muEF', 'muonSubtrFactor', 'neEmEF', 'neHEF', 'phi', 'pt', 'rawFactor', 'hadronFlavour', 'genJetIdx', 'partonFlavour']
recjetvars = ['chMultiplicity', 'jetId', 'nConstituents', 'nElectrons', 'nMuons', 'nSVs', 'neMultiplicity', 'electronIdx1', 'electronIdx2', 'muonIdx1', 'muonIdx2', 'svIdx1', 'svIdx2', 'hfadjacentEtaStripsSize', 'hfcentralEtaStripSize', 'PNetRegPtRawCorr', 'PNetRegPtRawCorrNeutrino', 'PNetRegPtRawRes', 'UParTAK4RegPtRawCorr', 'UParTAK4RegPtRawCorrNeutrino', 'UParTAK4RegPtRawRes', 'area', 'btagDeepFlavB', 'btagDeepFlavCvB', 'btagDeepFlavCvL', 'btagDeepFlavQG', 'btagPNetB', 'btagPNetCvB', 'btagPNetCvL', 'btagPNetCvNotB', 'btagPNetQvG', 'btagPNetTauVJet', 'btagUParTAK4B', 'btagUParTAK4CvB', 'btagUParTAK4CvL', 'btagUParTAK4CvNotB', 'btagUParTAK4QvG', 'btagUParTAK4TauVJet', 'chEmEF', 'chHEF', 'eta', 'hfEmEF', 'hfHEF', 'hfsigmaEtaEta', 'hfsigmaPhiPhi', 'mass', 'muEF', 'muonSubtrFactor', 'neEmEF', 'neHEF', 'phi', 'pt', 'rawFactor', 'hadronFlavour', 'genJetIdx', 'partonFlavour']
recjetvars = ['chMultiplicity', 'jetId', 'nConstituents', 'nElectrons', 'nMuons', 'nSVs', 'neMultiplicity', 'area', 'btagDeepFlavB', 'btagDeepFlavCvB', 'btagDeepFlavCvL', 'btagDeepFlavQG', 'btagPNetB', 'btagPNetCvB', 'btagPNetCvL', 'btagPNetCvNotB', 'btagPNetQvG', 'btagPNetTauVJet', 'btagUParTAK4B', 'btagUParTAK4CvB', 'btagUParTAK4CvL', 'btagUParTAK4CvNotB', 'btagUParTAK4QvG', 'btagUParTAK4TauVJet', 'chEmEF', 'chHEF', 'eta', 'hfEmEF', 'hfHEF', 'hfsigmaEtaEta', 'hfsigmaPhiPhi', 'mass', 'muEF', 'muonSubtrFactor', 'neEmEF', 'neHEF', 'phi', 'pt', 'rawFactor', 'hadronFlavour', 'genJetIdx', 'partonFlavour']
addrecjetvars = [
    'response',
    'event_nJet', 'event_nFatJet', 'event_PV_npvs', 'event_PV_npvsGood', 'event_MET_pt', 'event_MET_phi',
    'nearest_dR', 'nearest_pt',
    'nearestGen_dR', 'nearestGen_pt',
    'genMatch_dR', 'genMatch_eta', 'genMatch_mass', 'genMatch_phi', 'genMatch_pt', 'genMatch_partonFlavour', 'genMatch_hadronFlavour'
]

genfatjetvars = ['eta', 'mass', 'phi', 'pt', 'hadronFlavour', 'nBHadrons', 'nCHadrons', 'partonFlavour']
addgenfatjetvars = [
    'nearest_dR', 'nearest_pt'
]

# recfatjetvars = ['area', 'btagCSVV2', 'btagDDBvLV2', 'btagDDCvBV2', 'btagDDCvLV2', 'btagDeepB', 'btagHbb', 'electronIdx3SJ', 'eta', 'genJetAK8Idx', 'hadronFlavour', 'jetId', 'lsf3', 'mass', 'msoftdrop', 'muonIdx3SJ', 'n2b1', 'n3b1', 'nBHadrons', 'nCHadrons', 'nConstituents', 'particleNetMD_QCD', 'particleNetMD_Xbb', 'particleNetMD_Xcc', 'particleNetMD_Xqq', 'particleNet_H4qvsQCD', 'particleNet_HbbvsQCD', 'particleNet_HccvsQCD', 'particleNet_QCD', 'particleNet_TvsQCD', 'particleNet_WvsQCD', 'particleNet_ZvsQCD', 'particleNet_mass', 'phi', 'pt', 'rawFactor', 'subJetIdx1', 'subJetIdx2', 'tau1', 'tau2', 'tau3', 'tau4']
# recfatjetvars = ['area', 'btagCSVV2', 'btagDDBvLV2', 'btagDDCvBV2', 'btagDDCvLV2', 'btagDeepB', 'btagHbb', 'deepTagMD_H4qvsQCD', 'deepTagMD_HbbvsQCD', 'deepTagMD_TvsQCD', 'deepTagMD_WvsQCD', 'deepTagMD_ZHbbvsQCD', 'deepTagMD_ZHccvsQCD', 'deepTagMD_ZbbvsQCD', 'deepTagMD_ZvsQCD', 'deepTagMD_bbvsLight', 'deepTagMD_ccvsLight', 'deepTag_H', 'deepTag_QCD', 'deepTag_QCDothers', 'deepTag_TvsQCD', 'deepTag_WvsQCD', 'deepTag_ZvsQCD', 'electronIdx3SJ', 'eta', 'genJetAK8Idx', 'hadronFlavour', 'jetId', 'lsf3', 'mass', 'msoftdrop', 'muonIdx3SJ', 'n2b1', 'n3b1', 'nBHadrons', 'nCHadrons', 'nConstituents', 'particleNetMD_QCD', 'particleNetMD_Xbb', 'particleNetMD_Xcc', 'particleNetMD_Xqq', 'particleNet_H4qvsQCD', 'particleNet_HbbvsQCD', 'particleNet_HccvsQCD', 'particleNet_QCD', 'particleNet_TvsQCD', 'particleNet_WvsQCD', 'particleNet_ZvsQCD', 'particleNet_mass', 'phi', 'pt', 'rawFactor', 'subJetIdx1', 'subJetIdx2', 'tau1', 'tau2', 'tau3', 'tau4']
# recfatjetvars = ['jetId', 'nConstituents', 'area', 'btagDDBvLV2', 'btagDDCvBV2', 'btagDDCvLV2', 'btagDeepB', 'eta', 'mass', 'msoftdrop', 'n2b1', 'n3b1', 'particleNetWithMass_H4qvsQCD', 'particleNetWithMass_HbbvsQCD', 'particleNetWithMass_HccvsQCD', 'particleNetWithMass_QCD', 'particleNetWithMass_TvsQCD', 'particleNetWithMass_WvsQCD', 'particleNetWithMass_ZvsQCD', 'phi', 'pt', 'rawFactor', 'tau1', 'tau2', 'tau3', 'tau4', 'lsf3', 'hadronFlavour', 'nBHadrons', 'nCHadrons', 'genJetAK8Idx']
recfatjetvars = ['jetId', 'nConstituents', 'chMultiplicity', 'neMultiplicity', 'subJetIdx1', 'subJetIdx2', 'electronIdx3SJ', 'muonIdx3SJ', 'area', 'chEmEF', 'chHEF', 'eta', 'mass', 'msoftdrop', 'muEF', 'n2b1', 'n3b1', 'neEmEF', 'neHEF', 'particleNetLegacy_QCD', 'particleNetLegacy_Xbb', 'particleNetLegacy_Xcc', 'particleNetLegacy_Xqq', 'particleNetLegacy_mass', 'particleNetWithMass_H4qvsQCD', 'particleNetWithMass_HbbvsQCD', 'particleNetWithMass_HccvsQCD', 'particleNetWithMass_QCD', 'particleNetWithMass_TvsQCD', 'particleNetWithMass_WvsQCD', 'particleNetWithMass_ZvsQCD', 'particleNet_QCD', 'particleNet_QCD0HF', 'particleNet_QCD1HF', 'particleNet_QCD2HF', 'particleNet_XbbVsQCD', 'particleNet_XccVsQCD', 'particleNet_XggVsQCD', 'particleNet_XqqVsQCD', 'particleNet_XteVsQCD', 'particleNet_XtmVsQCD', 'particleNet_XttVsQCD', 'particleNet_massCorr', 'phi', 'pt', 'rawFactor', 'tau1', 'tau2', 'tau3', 'tau4', 'lsf3', 'hadronFlavour', 'genJetAK8Idx']
recfatjetvars = ['jetId', 'nConstituents', 'chMultiplicity', 'neMultiplicity', 'area', 'chEmEF', 'chHEF', 'eta', 'mass', 'msoftdrop', 'muEF', 'n2b1', 'n3b1', 'neEmEF', 'neHEF', 'particleNetLegacy_QCD', 'particleNetLegacy_Xbb', 'particleNetLegacy_Xcc', 'particleNetLegacy_Xqq', 'particleNetLegacy_mass', 'particleNetWithMass_H4qvsQCD', 'particleNetWithMass_HbbvsQCD', 'particleNetWithMass_HccvsQCD', 'particleNetWithMass_QCD', 'particleNetWithMass_TvsQCD', 'particleNetWithMass_WvsQCD', 'particleNetWithMass_ZvsQCD', 'particleNet_QCD', 'particleNet_QCD0HF', 'particleNet_QCD1HF', 'particleNet_QCD2HF', 'particleNet_XbbVsQCD', 'particleNet_XccVsQCD', 'particleNet_XggVsQCD', 'particleNet_XqqVsQCD', 'particleNet_XteVsQCD', 'particleNet_XtmVsQCD', 'particleNet_XttVsQCD', 'particleNet_massCorr', 'phi', 'pt', 'rawFactor', 'tau1', 'tau2', 'tau3', 'tau4', 'lsf3', 'hadronFlavour', 'genJetAK8Idx']
addrecfatjetvars = [
    'response',
    'event_nJet', 'event_nFatJet', 'event_PV_npvs', 'event_PV_npvsGood', 'event_MET_pt', 'event_MET_phi',
    'nearest_dR', 'nearest_pt',
    'nearestGen_dR', 'nearestGen_pt',
    'genMatch_dR', 'genMatch_eta', 'genMatch_mass', 'genMatch_phi', 'genMatch_pt', 'genMatch_partonFlavour', 'genMatch_hadronFlavour'
]

# not in run3 samples
# for rmv in ['bRegCorr', 'bRegRes', 'cRegCorr', 'cRegRes', 'chFPV0EF', 'cleanmask', 'puId', 'puIdDisc', 'qgl']:
#     recjetvars.remove(rmv)

cmsswversion = fastsimlist[0].split('/')[-6]
process = fastsimlist[0].split('/')[-4]
innstub = fastsimlist[0].split('/')[-1].replace('.root', '')
if isTest: outname = '/data/dust/user/wolfmor/Refinement/littletree_' + cmsswversion + '_' + process + '_' + innstub + '_coffea_PUPPI_new_test.root'
else: outname = '/data/dust/user/wolfmor/Refinement/littletree_' + cmsswversion + '_' + process + '_' + innstub + '_coffea_PUPPI_new.root'


def main():
    
    with uproot.recreate(outname) as outfile:


        # define event-level tree
        
        eventdict = {'EventID': float}
        geneventdict = {v: float for v in geneventvars}
        eventdict.update(geneventdict)
        fasteventdict = {v + '_FastSim': float for v in receventvars + addreceventvars}
        eventdict.update(fasteventdict)
        fulleventdict = {v + '_FullSim': float for v in receventvars + addreceventvars}
        eventdict.update(fulleventdict)

        outfile.mktree('tEvent', eventdict, title='tEvent')


        # define jet-level tree
        
        jetdict = {'EventID': float}
        gendict = {'GenJet_' + v: float for v in genjetvars + addgenjetvars}
        jetdict.update(gendict)
        fastdict = {'RecJet_' + v + '_FastSim': float for v in recjetvars + addrecjetvars}
        jetdict.update(fastdict)
        fulldict = {'RecJet_' + v + '_FullSim': float for v in recjetvars + addrecjetvars}
        jetdict.update(fulldict)

        outfile.mktree('tJet', jetdict, title='tJet')


        # define jet-level only FastSim tree

        jetonlyfastdict = {'RecJet_' + v + '_FastSim': float for v in recjetvars + addrecjetvars}

        outfile.mktree('tJetOnlyFast', jetonlyfastdict, title='tJetOnlyFast')


        # define fatjet-level tree
        
        fatjetdict = {'EventID': float}
        gendict = {'GenFatJet_' + v: float for v in genfatjetvars + addgenfatjetvars}
        fatjetdict.update(gendict)
        fastdict = {'RecFatJet_' + v + '_FastSim': float for v in recfatjetvars + addrecfatjetvars}
        fatjetdict.update(fastdict)
        fulldict = {'RecFatJet_' + v + '_FullSim': float for v in recfatjetvars + addrecfatjetvars}
        fatjetdict.update(fulldict)

        outfile.mktree('tFatJet', fatjetdict, title='tFatJet')


        # define fatjet-level only FastSim tree

        fatjetonlyfastdict = {'RecFatJet_' + v + '_FastSim': float for v in recfatjetvars + addrecfatjetvars}

        outfile.mktree('tFatJetOnlyFast', fatjetonlyfastdict, title='tFatJetOnlyFast')
        

        # loop over files
        
        total = 0
        fullsimnotexisting = 0
        filebroken = 0
        notsamenumevents = 0
        notsamenumgenjets = 0
        notsamegenjets = 0
        nogenjets = 0
        nofastjets = 0
        nofulljets = 0
        encounterednone = 0
        for ifname in range(len(fastsimlist)):

            if (isTest and nTest < 100) or ifname % 100 == 0: print(ifname, '/', len(fastsimlist))

            if isTest and nTest < 100: print(fastsimlist[ifname])

            if not os.path.exists(fullsimmodifier(fastsimlist[ifname])):
                if isTest and nTest < 100: print('fullsim doesnt exist yet')
                fullsimnotexisting += 1
                continue

            if os.path.getsize(fastsimlist[ifname]) < 10000 or os.path.getsize(fullsimmodifier(fastsimlist[ifname])) < 1000:
                if isTest and nTest < 100: print('file broken')
                filebroken += 1
                continue

            try:
                events_fast = NanoEventsFactory.from_root(fastsimlist[ifname],  # for 106: {fastsimlist[ifname]: 'Events'},
                                                          schemaclass=NanoAODSchema).events()
                events_full = NanoEventsFactory.from_root(fullsimmodifier(fastsimlist[ifname]),  # for 106: {fullsimmodifier(fastsimlist[ifname]): 'Events'},
                                                          schemaclass=NanoAODSchema).events()
            except uproot.exceptions.KeyInFileError:
                filebroken += 1
                continue

            if events_fast is None or events_full is None:
                filebroken += 1
                continue

            # TODO: uncomment?
            # if not len(events_fast) == len(events_full):
            #     print('not the same number of events in Fast/Full')
            #     notsamenumevents += 1
            #     continue

            if not ak.all(ak.num(events_fast.GenJet) == ak.num(events_full.GenJet)):
                print('not the same number of GenJets in Fast/Full')
                notsamenumgenjets += 1
                continue

            if not ak.all(events_fast.GenJet.pt == events_full.GenJet.pt):
                print('GenJets not the same')
                notsamegenjets += 1
                continue

            # if not ak.num(events_fast.GenJet) > 0:
            #     print('no GenJets')
            #     nogenjets += 1
            #     continue

            # if not ak.num(events_fast.Jet) > 0:
            #     print('no FastSim jets')
            #     nofastjets += 1
            #     continue

            # if not ak.num(events_full.Jet) > 0:
            #     print('no FullSim jets')
            #     nofulljets += 1
            #     continue

            # skip events with no fastsim jets
            events_fast = events_fast[ak.num(events_fast.Jet) > 0]
            events_full = events_full[ak.num(events_fast.Jet) > 0]

            allfastjets = events_fast.Jet
            allfulljets = events_full.Jet
            allgenjets = events_fast.GenJet
            
            allfastfatjets = events_fast.FatJet
            allfullfatjets = events_full.FatJet
            allgenfatjets = events_fast.GenJetAK8


            # define additional variables

            allfastjets['EventID'] = 10000 * ifname + events_fast.event

            allfastjets['event_nJet'] = ak.num(events_fast.Jet)
            allfastjets['event_nFatJet'] = ak.num(events_fast.FatJet)
            allfastjets['event_PV_npvs'] = events_fast.PV.npvs
            allfastjets['event_PV_npvsGood'] = events_fast.PV.npvsGood
            allfastjets['event_MET_pt'] = events_fast.PuppiMET.pt
            allfastjets['event_MET_phi'] = events_fast.PuppiMET.phi
            
            allfastjets_nearest, allfastjets_nearest_dR = allfastjets.nearest(allfastjets, metric=lambda a, b: ak.where(a.delta_r(b) == 0, 9, a.delta_r(b)), return_metric=True)
            allfastjets['nearest_dR'] = allfastjets_nearest_dR
            allfastjets['nearest_pt'] = allfastjets_nearest.pt
            allfastjets_nearestGen, allfastjets_nearestGen_dR = allfastjets.nearest(allgenjets, return_metric=True)
            allfastjets['nearestGen_dR'] = allfastjets_nearestGen_dR
            allfastjets['nearestGen_pt'] = allfastjets_nearestGen.pt

            allfulljets['event_nJet'] = ak.num(events_full.Jet)
            allfulljets['event_nFatJet'] = ak.num(events_full.FatJet)
            allfulljets['event_PV_npvs'] = events_full.PV.npvs
            allfulljets['event_PV_npvsGood'] = events_full.PV.npvsGood
            allfulljets['event_MET_pt'] = events_full.PuppiMET.pt
            allfulljets['event_MET_phi'] = events_full.PuppiMET.phi
            
            allfulljets_nearest, allfulljets_nearest_dR = allfulljets.nearest(allfulljets, metric=lambda a, b: ak.where(a.delta_r(b) == 0, 9, a.delta_r(b)), return_metric=True)
            allfulljets['nearest_dR'] = allfulljets_nearest_dR
            allfulljets['nearest_pt'] = allfulljets_nearest.pt
            allfulljets_nearestGen, allfulljets_nearestGen_dR = allfulljets.nearest(allgenjets, return_metric=True)
            allfulljets['nearestGen_dR'] = allfulljets_nearestGen_dR
            allfulljets['nearestGen_pt'] = allfulljets_nearestGen.pt
            
            
            allfastfatjets['EventID'] = 10000 * ifname + events_fast.event
            
            allfastfatjets['event_nJet'] = ak.num(events_fast.Jet)
            allfastfatjets['event_nFatJet'] = ak.num(events_fast.FatJet)
            allfastfatjets['event_PV_npvs'] = events_fast.PV.npvs
            allfastfatjets['event_PV_npvsGood'] = events_fast.PV.npvsGood
            allfastfatjets['event_MET_pt'] = events_fast.PuppiMET.pt
            allfastfatjets['event_MET_phi'] = events_fast.PuppiMET.phi
            
            allfastfatjets_nearest, allfastfatjets_nearest_dR = allfastfatjets.nearest(allfastfatjets, metric=lambda a, b: ak.where(a.delta_r(b) == 0, 9, a.delta_r(b)), return_metric=True)
            allfastfatjets['nearest_dR'] = allfastfatjets_nearest_dR
            allfastfatjets['nearest_pt'] = allfastfatjets_nearest.pt
            allfastfatjets_nearestGen, allfastfatjets_nearestGen_dR = allfastfatjets.nearest(allgenfatjets, return_metric=True)
            allfastfatjets['nearestGen_dR'] = allfastfatjets_nearestGen_dR
            allfastfatjets['nearestGen_pt'] = allfastfatjets_nearestGen.pt
            
            allfullfatjets['event_nJet'] = ak.num(events_full.Jet)
            allfullfatjets['event_nFatJet'] = ak.num(events_full.FatJet)
            allfullfatjets['event_PV_npvs'] = events_full.PV.npvs
            allfullfatjets['event_PV_npvsGood'] = events_full.PV.npvsGood
            allfullfatjets['event_MET_pt'] = events_full.PuppiMET.pt
            allfullfatjets['event_MET_phi'] = events_full.PuppiMET.phi
            
            allfullfatjets_nearest, allfullfatjets_nearest_dR = allfullfatjets.nearest(allfullfatjets, metric=lambda a, b: ak.where(a.delta_r(b) == 0, 9, a.delta_r(b)), return_metric=True)
            allfullfatjets['nearest_dR'] = allfullfatjets_nearest_dR
            allfullfatjets['nearest_pt'] = allfullfatjets_nearest.pt
            allfullfatjets_nearestGen, allfullfatjets_nearestGen_dR = allfullfatjets.nearest(allgenfatjets, return_metric=True)
            allfullfatjets['nearestGen_dR'] = allfullfatjets_nearestGen_dR
            allfullfatjets['nearestGen_pt'] = allfullfatjets_nearestGen.pt
            

            # fill event-level tree
            
            eventdict = {'EventID': allfastjets.EventID[:, 0]}

            geneventdict = {v: getattr(getattr(events_fast, v.split('_')[0]), v.split('_')[1]) for v in geneventvars if not v.startswith('n')}
            geneventdict.update({v: ak.num(getattr(events_fast, v[1:])) for v in geneventvars if v.startswith('n')})
            eventdict.update(geneventdict)

            # TODO: add unrefined ?
            # fasteventdict = {v + '_FastSim': getattr(getattr(events_fast, v.split('_')[0]), v.split('_')[1] + ('unrefined' if 'btagDeepFlav' in v.split('_')[1] else '')) for v in receventvars if not v.startswith('n')}
            fasteventdict = {v + '_FastSim': getattr(getattr(events_fast, v.split('_')[0]), v.split('_')[1]) for v in receventvars if not v.startswith('n')}
            fasteventdict.update({v + '_FastSim': ak.num(getattr(events_fast, v[1:])) for v in receventvars if v.startswith('n')})
            # TODO: uncomment (M)HT?
            # fasteventdict.update({'HT_FastSim': ak.sum(events_fast.Jet.pt[(events_fast.Jet.pt > 15) & (abs(events_fast.Jet.eta) < 2.4)], axis=1)})
            # fastmht = events_fast.Jet[(events_fast.Jet.pt > 15) & (abs(events_fast.Jet.eta) < 5)].sum(axis=1)
            # fasteventdict.update({'MHT_pt_FastSim': fastmht.pt})
            # fasteventdict.update({'MHT_phi_FastSim': fastmht.phi})
            eventdict.update(fasteventdict)

            fulleventdict = {v + '_FullSim': getattr(getattr(events_full, v.split('_')[0]), v.split('_')[1]) for v in receventvars if not v.startswith('n')}
            fulleventdict.update({v + '_FullSim': ak.num(getattr(events_full, v[1:])) for v in receventvars if v.startswith('n')})
            # TODO: uncomment (M)HT?
            # fulleventdict.update({'HT_FullSim': ak.sum(events_full.Jet.pt[(events_full.Jet.pt > 15) & (abs(events_full.Jet.eta) < 2.4)], axis=1)})
            # fullmht = events_full.Jet[(events_full.Jet.pt > 15) & (abs(events_full.Jet.eta) < 5)].sum(axis=1)
            # fulleventdict.update({'MHT_pt_FullSim': fullmht.pt})
            # fulleventdict.update({'MHT_phi_FullSim': fullmht.phi})
            eventdict.update(fulleventdict)


            # find matching jet triplets

            fastjets = allfastjets[allfastjets.genJetIdx > -1]

            fulljets, mask = fastjets.nearest(allfulljets[allfulljets.genJetIdx > -1], metric=lambda a, b: a.matched_gen.delta_r(b.matched_gen), return_metric=True)
            mask = ak.fill_none(mask, -1)

            fastjets = fastjets[mask == 0]
            fulljets = fulljets[mask == 0]
            genjets = fastjets.matched_gen
            
            
            # find matching fatjet triplets

            fastfatjets = allfastfatjets[allfastfatjets.genJetAK8Idx > -1]

            fullfatjets, mask = fastfatjets.nearest(allfullfatjets[allfullfatjets.genJetAK8Idx > -1], metric=lambda a, b: a.matched_gen.delta_r(b.matched_gen), return_metric=True)
            mask = ak.fill_none(mask, -1)

            fastfatjets = fastfatjets[mask == 0]
            fullfatjets = fullfatjets[mask == 0]
            genfatjets = fastfatjets.matched_gen


            # define additional variables for genjets
            allgenjets_nearest, allgenjets_nearest_dR = genjets.nearest(allgenjets, metric=lambda a, b: ak.where(a.delta_r(b) == 0, 9, a.delta_r(b)), return_metric=True)
            genjets['nearest_dR'] = allgenjets_nearest_dR
            genjets['nearest_pt'] = allgenjets_nearest.pt
            
            
            # define additional variables for genfatjets
            allgenfatjets_nearest, allgenfatjets_nearest_dR = genfatjets.nearest(allgenfatjets, metric=lambda a, b: ak.where(a.delta_r(b) == 0, 9, a.delta_r(b)), return_metric=True)
            genfatjets['nearest_dR'] = allgenfatjets_nearest_dR
            genfatjets['nearest_pt'] = allgenfatjets_nearest.pt


            # flatten the arrays
            allfastjets = ak.flatten(allfastjets)
            fastjets = ak.flatten(fastjets)
            fulljets = ak.flatten(fulljets)
            genjets = ak.flatten(genjets)
            allfastfatjets = ak.flatten(allfastfatjets)
            fastfatjets = ak.flatten(fastfatjets)
            fullfatjets = ak.flatten(fullfatjets)
            genfatjets = ak.flatten(genfatjets)


            # fill jet-level tree

            jetdict = {'EventID': fastjets.EventID}

            gendict = {'GenJet_' + v: getattr(genjets, v) for v in genjetvars}
            gendict.update({'GenJet_nearest_dR': genjets.nearest_dR})
            gendict.update({'GenJet_nearest_pt': genjets.nearest_pt})
            jetdict.update(gendict)
            
            fastdict = {'RecJet_' + v + '_FastSim': getattr(fastjets, v) for v in recjetvars}
            fastdict.update({'RecJet_response_FastSim': fastjets.pt / genjets.pt})
            fastdict.update({'RecJet_event_nJet_FastSim': fastjets.event_nJet})
            fastdict.update({'RecJet_event_nFatJet_FastSim': fastjets.event_nFatJet})
            fastdict.update({'RecJet_event_PV_npvs_FastSim': fastjets.event_PV_npvs})
            fastdict.update({'RecJet_event_PV_npvsGood_FastSim': fastjets.event_PV_npvsGood})
            fastdict.update({'RecJet_event_MET_pt_FastSim': fastjets.event_MET_pt})
            fastdict.update({'RecJet_event_MET_phi_FastSim': fastjets.event_MET_phi})
            fastdict.update({'RecJet_nearest_dR_FastSim': fastjets.nearest_dR})
            fastdict.update({'RecJet_nearest_pt_FastSim': fastjets.nearest_pt})
            fastdict.update({'RecJet_nearestGen_dR_FastSim': fastjets.nearestGen_dR})
            fastdict.update({'RecJet_nearestGen_pt_FastSim': fastjets.nearestGen_pt})
            fastdict.update({'RecJet_genMatch_dR_FastSim': fastjets.delta_r(fastjets.matched_gen)})
            fastdict.update({'RecJet_genMatch_eta_FastSim': fastjets.matched_gen.eta})
            fastdict.update({'RecJet_genMatch_mass_FastSim': fastjets.matched_gen.mass})
            fastdict.update({'RecJet_genMatch_phi_FastSim': fastjets.matched_gen.phi})
            fastdict.update({'RecJet_genMatch_pt_FastSim': fastjets.matched_gen.pt})
            fastdict.update({'RecJet_genMatch_partonFlavour_FastSim': fastjets.matched_gen.partonFlavour})
            fastdict.update({'RecJet_genMatch_hadronFlavour_FastSim': fastjets.matched_gen.hadronFlavour})
            jetdict.update(fastdict)
            
            fulldict = {'RecJet_' + v + '_FullSim': getattr(fulljets, v) for v in recjetvars}
            fulldict.update({'RecJet_response_FullSim': fulljets.pt / genjets.pt})
            fulldict.update({'RecJet_event_nJet_FullSim': fulljets.event_nJet})
            fulldict.update({'RecJet_event_nFatJet_FullSim': fulljets.event_nFatJet})
            fulldict.update({'RecJet_event_PV_npvs_FullSim': fulljets.event_PV_npvs})
            fulldict.update({'RecJet_event_PV_npvsGood_FullSim': fulljets.event_PV_npvsGood})
            fulldict.update({'RecJet_event_MET_pt_FullSim': fulljets.event_MET_pt})
            fulldict.update({'RecJet_event_MET_phi_FullSim': fulljets.event_MET_phi})
            fulldict.update({'RecJet_nearest_dR_FullSim': fulljets.nearest_dR})
            fulldict.update({'RecJet_nearest_pt_FullSim': fulljets.nearest_pt})
            fulldict.update({'RecJet_nearestGen_dR_FullSim': fulljets.nearestGen_dR})
            fulldict.update({'RecJet_nearestGen_pt_FullSim': fulljets.nearestGen_pt})
            fulldict.update({'RecJet_genMatch_dR_FullSim': fulljets.delta_r(fulljets.matched_gen)})
            fulldict.update({'RecJet_genMatch_eta_FullSim': fulljets.matched_gen.eta})
            fulldict.update({'RecJet_genMatch_mass_FullSim': fulljets.matched_gen.mass})
            fulldict.update({'RecJet_genMatch_phi_FullSim': fulljets.matched_gen.phi})
            fulldict.update({'RecJet_genMatch_pt_FullSim': fulljets.matched_gen.pt})
            fulldict.update({'RecJet_genMatch_partonFlavour_FullSim': fulljets.matched_gen.partonFlavour})
            fulldict.update({'RecJet_genMatch_hadronFlavour_FullSim': fulljets.matched_gen.hadronFlavour})
            jetdict.update(fulldict)


            # fill jet-level only FastSim tree

            jetonlyfastdict = {'RecJet_' + v + '_FastSim': getattr(allfastjets, v) for v in recjetvars}
            jetonlyfastdict.update({'RecJet_response_FastSim': allfastjets.pt / allfastjets.matched_gen.pt})
            jetonlyfastdict.update({'RecJet_event_nJet_FastSim': allfastjets.event_nJet})
            jetonlyfastdict.update({'RecJet_event_nFatJet_FastSim': allfastjets.event_nFatJet})
            jetonlyfastdict.update({'RecJet_event_PV_npvs_FastSim': allfastjets.event_PV_npvs})
            jetonlyfastdict.update({'RecJet_event_PV_npvsGood_FastSim': allfastjets.event_PV_npvsGood})
            jetonlyfastdict.update({'RecJet_event_MET_pt_FastSim': allfastjets.event_MET_pt})
            jetonlyfastdict.update({'RecJet_event_MET_phi_FastSim': allfastjets.event_MET_phi})
            jetonlyfastdict.update({'RecJet_nearest_dR_FastSim': allfastjets.nearest_dR})
            jetonlyfastdict.update({'RecJet_nearest_pt_FastSim': allfastjets.nearest_pt})
            jetonlyfastdict.update({'RecJet_nearestGen_dR_FastSim': allfastjets.nearestGen_dR})
            jetonlyfastdict.update({'RecJet_nearestGen_pt_FastSim': allfastjets.nearestGen_pt})
            jetonlyfastdict.update({'RecJet_genMatch_dR_FastSim': allfastjets.delta_r(allfastjets.matched_gen)})
            jetonlyfastdict.update({'RecJet_genMatch_eta_FastSim': allfastjets.matched_gen.eta})
            jetonlyfastdict.update({'RecJet_genMatch_mass_FastSim': allfastjets.matched_gen.mass})
            jetonlyfastdict.update({'RecJet_genMatch_phi_FastSim': allfastjets.matched_gen.phi})
            jetonlyfastdict.update({'RecJet_genMatch_pt_FastSim': allfastjets.matched_gen.pt})
            jetonlyfastdict.update({'RecJet_genMatch_partonFlavour_FastSim': allfastjets.matched_gen.partonFlavour})
            jetonlyfastdict.update({'RecJet_genMatch_hadronFlavour_FastSim': allfastjets.matched_gen.hadronFlavour})
            
            
            # fill fatjet-level tree

            fatjetdict = {'EventID': fastfatjets.EventID}

            gendict = {'GenFatJet_' + v: getattr(genfatjets, v) for v in genfatjetvars}
            gendict.update({'GenFatJet_nearest_dR': genfatjets.nearest_dR})
            gendict.update({'GenFatJet_nearest_pt': genfatjets.nearest_pt})
            fatjetdict.update(gendict)

            fastdict = {'RecFatJet_' + v + '_FastSim': getattr(fastfatjets, v) for v in recfatjetvars}
            fastdict.update({'RecFatJet_response_FastSim': fastfatjets.pt / genfatjets.pt})
            fastdict.update({'RecFatJet_event_nJet_FastSim': fastfatjets.event_nJet})
            fastdict.update({'RecFatJet_event_nFatJet_FastSim': fastfatjets.event_nFatJet})
            fastdict.update({'RecFatJet_event_PV_npvs_FastSim': fastfatjets.event_PV_npvs})
            fastdict.update({'RecFatJet_event_PV_npvsGood_FastSim': fastfatjets.event_PV_npvsGood})
            fastdict.update({'RecFatJet_event_MET_pt_FastSim': fastfatjets.event_MET_pt})
            fastdict.update({'RecFatJet_event_MET_phi_FastSim': fastfatjets.event_MET_phi})
            fastdict.update({'RecFatJet_nearest_dR_FastSim': fastfatjets.nearest_dR})
            fastdict.update({'RecFatJet_nearest_pt_FastSim': fastfatjets.nearest_pt})
            fastdict.update({'RecFatJet_nearestGen_dR_FastSim': fastfatjets.nearestGen_dR})
            fastdict.update({'RecFatJet_nearestGen_pt_FastSim': fastfatjets.nearestGen_pt})
            fastdict.update({'RecFatJet_genMatch_dR_FastSim': fastfatjets.delta_r(fastfatjets.matched_gen)})
            fastdict.update({'RecFatJet_genMatch_eta_FastSim': fastfatjets.matched_gen.eta})
            fastdict.update({'RecFatJet_genMatch_mass_FastSim': fastfatjets.matched_gen.mass})
            fastdict.update({'RecFatJet_genMatch_phi_FastSim': fastfatjets.matched_gen.phi})
            fastdict.update({'RecFatJet_genMatch_pt_FastSim': fastfatjets.matched_gen.pt})
            fastdict.update({'RecFatJet_genMatch_partonFlavour_FastSim': fastfatjets.matched_gen.partonFlavour})
            fastdict.update({'RecFatJet_genMatch_hadronFlavour_FastSim': fastfatjets.matched_gen.hadronFlavour})
            fatjetdict.update(fastdict)
            
            fulldict = {'RecFatJet_' + v + '_FullSim': getattr(fullfatjets, v) for v in recfatjetvars}
            fulldict.update({'RecFatJet_response_FullSim': fullfatjets.pt / genfatjets.pt})
            fulldict.update({'RecFatJet_event_nJet_FullSim': fullfatjets.event_nJet})
            fulldict.update({'RecFatJet_event_nFatJet_FullSim': fullfatjets.event_nFatJet})
            fulldict.update({'RecFatJet_event_PV_npvs_FullSim': fullfatjets.event_PV_npvs})
            fulldict.update({'RecFatJet_event_PV_npvsGood_FullSim': fullfatjets.event_PV_npvsGood})
            fulldict.update({'RecFatJet_event_MET_pt_FullSim': fullfatjets.event_MET_pt})
            fulldict.update({'RecFatJet_event_MET_phi_FullSim': fullfatjets.event_MET_phi})
            fulldict.update({'RecFatJet_nearest_dR_FullSim': fullfatjets.nearest_dR})
            fulldict.update({'RecFatJet_nearest_pt_FullSim': fullfatjets.nearest_pt})
            fulldict.update({'RecFatJet_nearestGen_dR_FullSim': fullfatjets.nearestGen_dR})
            fulldict.update({'RecFatJet_nearestGen_pt_FullSim': fullfatjets.nearestGen_pt})
            fulldict.update({'RecFatJet_genMatch_dR_FullSim': fullfatjets.delta_r(fullfatjets.matched_gen)})
            fulldict.update({'RecFatJet_genMatch_eta_FullSim': fullfatjets.matched_gen.eta})
            fulldict.update({'RecFatJet_genMatch_mass_FullSim': fullfatjets.matched_gen.mass})
            fulldict.update({'RecFatJet_genMatch_phi_FullSim': fullfatjets.matched_gen.phi})
            fulldict.update({'RecFatJet_genMatch_pt_FullSim': fullfatjets.matched_gen.pt})
            fulldict.update({'RecFatJet_genMatch_partonFlavour_FullSim': fullfatjets.matched_gen.partonFlavour})
            fulldict.update({'RecFatJet_genMatch_hadronFlavour_FullSim': fullfatjets.matched_gen.hadronFlavour})
            fatjetdict.update(fulldict)


            # fill fatjet-level only FastSim tree
            
            fatjetonlyfastdict = {'RecFatJet_' + v + '_FastSim': getattr(allfastfatjets, v) for v in recfatjetvars}
            fatjetonlyfastdict.update({'RecFatJet_response_FastSim': allfastfatjets.pt / allfastfatjets.matched_gen.pt})
            fatjetonlyfastdict.update({'RecFatJet_event_nJet_FastSim': allfastfatjets.event_nJet})
            fatjetonlyfastdict.update({'RecFatJet_event_nFatJet_FastSim': allfastfatjets.event_nFatJet})
            fatjetonlyfastdict.update({'RecFatJet_event_PV_npvs_FastSim': allfastfatjets.event_PV_npvs})
            fatjetonlyfastdict.update({'RecFatJet_event_PV_npvsGood_FastSim': allfastfatjets.event_PV_npvsGood})
            fatjetonlyfastdict.update({'RecFatJet_event_MET_pt_FastSim': allfastfatjets.event_MET_pt})
            fatjetonlyfastdict.update({'RecFatJet_event_MET_phi_FastSim': allfastfatjets.event_MET_phi})
            fatjetonlyfastdict.update({'RecFatJet_nearest_dR_FastSim': allfastfatjets.nearest_dR})
            fatjetonlyfastdict.update({'RecFatJet_nearest_pt_FastSim': allfastfatjets.nearest_pt})
            fatjetonlyfastdict.update({'RecFatJet_nearestGen_dR_FastSim': allfastfatjets.nearestGen_dR})
            fatjetonlyfastdict.update({'RecFatJet_nearestGen_pt_FastSim': allfastfatjets.nearestGen_pt})
            fatjetonlyfastdict.update({'RecFatJet_genMatch_dR_FastSim': allfastfatjets.delta_r(allfastfatjets.matched_gen)})
            fatjetonlyfastdict.update({'RecFatJet_genMatch_eta_FastSim': allfastfatjets.matched_gen.eta})
            fatjetonlyfastdict.update({'RecFatJet_genMatch_mass_FastSim': allfastfatjets.matched_gen.mass})
            fatjetonlyfastdict.update({'RecFatJet_genMatch_phi_FastSim': allfastfatjets.matched_gen.phi})
            fatjetonlyfastdict.update({'RecFatJet_genMatch_pt_FastSim': allfastfatjets.matched_gen.pt})
            fatjetonlyfastdict.update({'RecFatJet_genMatch_partonFlavour_FastSim': allfastfatjets.matched_gen.partonFlavour})
            fatjetonlyfastdict.update({'RecFatJet_genMatch_hadronFlavour_FastSim': allfastfatjets.matched_gen.hadronFlavour})


            for _dict in [eventdict, jetdict, jetonlyfastdict, fatjetdict, fatjetonlyfastdict]:
                for key in _dict:
                    _dict[key] = ak.fill_none(_dict[key], -1)  # clean None values
                    # _dict[key] = _dict[key].compute()  # for 106 to "convert" dask arrays to real values

            # fill trees
            outfile['tEvent'].extend(eventdict)
            outfile['tJet'].extend(jetdict)
            outfile['tJetOnlyFast'].extend(jetonlyfastdict)
            outfile['tFatJet'].extend(fatjetdict)
            outfile['tFatJetOnlyFast'].extend(fatjetonlyfastdict)


            total += 1

    print('')
    print('just created', outname)
    print('total', total)
    print('fullsimnotexisting', fullsimnotexisting)
    print('filebroken', filebroken)
    print('notsamenumevents', notsamenumevents)
    print('notsamenumgenjets', notsamenumgenjets)
    print('notsamegenjets', notsamegenjets)
    print('nofastjets', nofastjets)
    print('encounterednone', encounterednone)


main()
