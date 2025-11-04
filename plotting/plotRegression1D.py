#! /usr/bin/env python


import ROOT
from plotting import Plotting
from array import array
import os, sys

normalize = True

try: training_id = sys.argv[1]
except: 
    training_id = '7April2025'
    training_id = '20231105_1'
    training_id = '20250407'
    training_id = '7April2025withClassifier'
    training_id = '7April2025'
    
#fname = '/data/dust/user/wolfmor/Refinement/TrainingOutput/output_refinement_regression_TRAININGID.root'
fname = '/data/dust/user/beinsam/FastSim/Refinement/Regress/TrainingOutput/output_refinement_regression_TRAININGID.root'

fin = ROOT.TFile(fname.replace('TRAININGID', training_id))
tree = fin.Get('tJet')

if not os.path.exists('figs'+training_id):
    os.system('mkdir figs'+training_id)
nameout = 'reg1D' + '_VARIABLE.png'

presel = 'RecJet_btagUParTAK4B_FastSim>0&&RecJet_btagUParTAK4B_FullSim>0'
# (name, suffix, label, color, fillstyle, linewidth, selection, legend option)
samples = [
    ('full', 'FullSim', 'FullSim', ROOT.kGreen+2, 3004, 3, presel+' && ', 'lf'),
    ('fast', 'FastSim', 'FastSim', ROOT.kRed+1, 0, 3, presel+' && ', 'l'),
    # ('refined', 'Refined', 'FastSim Refined', ROOT.kAzure+2, 0, 3, '1&&', 'l'),
    ('refined', 'Refined', 'FastSim Refined', ROOT.kAzure+2, 0, 3, presel+' && isTrainValTest<2 &&', 'l'),
    ('refinedtest', 'Refined', 'FastSim Refined (Test)', ROOT.kAzure+1, 0, 3, presel+' && isTrainValTest>1&&', 'lp'),
]

# (name, branch name, (nbins, xlow, xhigh), title, ratio plot range, selection)
##variables = [
##    ('DeepFlavB', 'RecJet_btagDeepFlavB_CLASS', (60, -0.1, 1.1), 'DeepJet b+bb+lepb Discriminator', 0.3, '1')
##]
nbins = 60
#nbins = 20
variables = [
    ('JetPt',     'RecJet_pt_CLASS', (nbins, 0, 2000),'Jet p_{T} [GeV]', 0.3, '1'),
    ('DeepFlavB', 'RecJet_btagDeepFlavB_CLASS', (nbins, -0.1, 1.1),'DeepJet b+bb+lepb Discriminator', 0.3, '1'),
    ('DeepFlavCvB', 'RecJet_btagDeepFlavCvB_CLASS', (nbins, -0.1, 1.1),'DeepJet C/B Discriminator', 0.3, '1'),
    ('DeepFlavCvL', 'RecJet_btagDeepFlavCvL_CLASS', (nbins, -0.1, 1.1),'DeepJet C/L Discriminator', 0.3, '1'),
    ('DeepFlavQG', 'RecJet_btagDeepFlavQG_CLASS', (nbins, -0.1, 1.1),'DeepJet Q/g Discriminator', 0.3, '1'),
    ('UParTAK4B', 'RecJet_btagUParTAK4B_CLASS', (nbins, -0.1, 1.1),'UParTAK4 B Discriminator', 0.3, '1'),
    ('UParTAK4CvB', 'RecJet_btagUParTAK4CvB_CLASS', (nbins, -0.1, 1.1),'UParTAK4 C/B Discriminator', 0.3, '1'),
    ('UParTAK4CvL', 'RecJet_btagUParTAK4CvL_CLASS', (nbins, -0.1, 1.1),'UParTAK4 C/L Discriminator', 0.3, '1'),
    ('UParTAK4QG', 'RecJet_btagUParTAK4QvG_CLASS', (nbins, -0.1, 1.1),'UParTAK4 Q/g Discriminator', 0.3, '1'),
    
]

presel = " RecJet_btagUParTAK4B_FastSim>0 && RecJet_btagUParTAK4B_FullSim>0 && "
presel = ""
# (name, suffix, label, color, fillstyle, linewidth, selection, legend option)
samples = [
    ('full', 'FullSim', 'FullSim', ROOT.kGreen+2, 3004, 3, presel, 'lf'),
    ('fast', 'FastSim', 'FastSim', ROOT.kRed+1, 0, 3, presel, 'l'),
    # ('refined', 'Refined', 'FastSim Refined', ROOT.kAzure+2, 0, 3, '1&&', 'l'),
    ('refined', 'Refined', 'FastSim Refined', ROOT.kAzure+2, 0, 3, presel+'isTrainValTest<2 && ', 'l'),
    ('refinedtest', 'Refined', 'FastSim Refined (Test)', ROOT.kAzure+1, 0, 3, presel+'isTrainValTest>1 && ', 'lp'),
]



if 'wClsfr' in training_id: variables.append(('Z classifier', 'RecJet_ffClassifier_CLASS', (nbins, -0.1, 1.1),'Z(Fast-Full classifier)', 0.3, '1'))

height = 800
width = 800

p = Plotting(
    text='(13.6 TeV)',
    extratext='Simulation',
    H_ref=height,
    W_ref=width,
    iPos=0
)

p.setStyle()

c0 = ROOT.TCanvas('c0', 'c0', 1, 1)

histos = {}
ratios = {}
canvas = {}
subpads = {}
lines = {}
leg = {}
for var in variables:

    print(var[0])

    c0.cd()

    for sam in samples:
        hstub = var[0] + sam[0]
        if type(var[2]) == list:
            histos[hstub] = ROOT.TH1F('h' + hstub, '', len(var[2])-1, array('d', var[2]))
        else:
            histos[hstub] = ROOT.TH1F('h' + hstub, '', var[2][0], var[2][1], var[2][2])

        tree.Draw(var[1].replace('CLASS', sam[1]) + '>>h' + hstub, sam[6] + var[5].replace('CLASS', sam[1]), '')

        if normalize: histos[hstub].Scale(1. / histos[hstub].Integral())

        histos[hstub].SetFillStyle(sam[4])
        histos[hstub].SetFillColor(sam[3])
        histos[hstub].SetLineWidth(sam[5])
        if 'test' not in sam[0] and 'val' not in sam[0]: histos[hstub].SetMarkerSize(0)
        histos[hstub].SetLineColor(sam[3])
        histos[hstub].SetMarkerColor(sam[3])

    for sam in samples:

        if sam[0] == 'full': continue
        hstub = var[0] + sam[0]
        if not histos[hstub].GetSumw2N(): histos[hstub].Sumw2()
        ratios[hstub] = histos[hstub].Clone('hr' + hstub)
        ratios[hstub].UseCurrentStyle()
        ratios[hstub].SetStats(0)
        ratios[hstub].Divide(histos[var[0] + 'full'])
        ratios[hstub].SetFillStyle(sam[4])
        ratios[hstub].SetFillColor(sam[3])
        ratios[hstub].SetLineWidth(sam[5])
        ratios[hstub].SetMarkerSize(0)
        ratios[hstub].SetLineColor(sam[3])
        ratios[hstub].SetMarkerColor(sam[3])


    leg[var[0]] = ROOT.TLegend(0.5, 0.6, 0.95, 0.9)

    canvas[var[0]] = ROOT.TCanvas('c' + var[0], 'c' + var[0], 2*width, height)
    canvas[var[0]].Divide(2, 1)

    subpads[var[0]] = p.addLowerPads(canvas[var[0]])

    subpads[var[0]]['1_1'].cd()

    for sam in samples:
        if 'test' in sam[0] or 'val' in sam[0]: histos[var[0] + sam[0]].Draw('e0 x0 same')
        else: histos[var[0] + sam[0]].Draw('hist same')
        leg[var[0]].AddEntry(histos[var[0] + sam[0]], sam[2], sam[7] if len(sam) > 7 else 'lpf')

    histos[var[0] + samples[0][0]].SetMaximum(1.6*max([histos[key].GetMaximum() for key in histos if var[0] == key[:len(var[0])] and key[len(var[0]):] in [sam[0] for sam in samples]]))
    histos[var[0] + samples[0][0]].GetXaxis().SetTitle(var[3])
    histos[var[0] + samples[0][0]].GetXaxis().SetLabelSize(0)
    if normalize: histos[var[0] + samples[0][0]].GetYaxis().SetTitle('Fraction of Jets')
    else: histos[var[0] + samples[0][0]].GetYaxis().SetTitle('Jets')

    leg[var[0]].Draw('same')

    p.postparePad()

    subpads[var[0]]['2_1'].cd()

    histos[var[0] + 'emptyloghist'] = histos[var[0] + samples[0][0]].Clone('emptyloghist')
    histos[var[0] + 'emptyloghist'].Reset()
    histos[var[0] + 'emptyloghist'].Draw('AXIS')

    for sam in samples:
        hstub = var[0] + sam[0]
        if 'test' in sam[0] or 'val' in sam[0]: histos[hstub].Draw('e0 x0 same')
        else: histos[hstub].Draw('hist same')

    #globalmin = min([histos[key].GetMinimum(0) for key in histos if var[0] == key[:len(var[0])] and key[len(var[0]):] in [sam[0] for sam in samples]])
    #globalmax = min([histos[key].GetMaximum() for key in histos if var[0] == key[:len(var[0])] and key[len(var[0]):] in [sam[0] for sam in samples]])
    
    globalmin = histos[var[0] + samples[0][0]].GetMinimum(0)
    globalmax = histos[var[0] + samples[0][0]].GetMaximum()
        
    
    logrange = ROOT.TMath.Log10(globalmax) - ROOT.TMath.Log10(globalmin)

    histos[var[0] + 'emptyloghist'].SetMinimum(0.5 * globalmin)
    histos[var[0] + 'emptyloghist'].SetMaximum(globalmax * 10 ** max(1, logrange))
    histos[var[0] + 'emptyloghist'].GetXaxis().SetTitle(var[3])
    if normalize: histos[var[0] + 'emptyloghist'].GetYaxis().SetTitle('Fraction of Jets')
    else: histos[var[0] + 'emptyloghist'].GetYaxis().SetTitle('Jets')

    leg[var[0]].Draw('SAME')

    ROOT.gPad.SetLogy()

    p.postparePad()

    for padname in ['1_2', '2_2']:

        subpads[var[0]][padname].cd()

        firstname = None
        for sam in samples:
            hstub = var[0] + sam[0]
            if sam[0] == 'full': continue
            if firstname is None: firstname = sam[0]
            if 'test' in sam[0] or 'val' in sam[0]:
                ratios[hstub].Draw('e0 x0 same')
                ratios[hstub].Draw('hist same')
            else:
                ratios[hstub].Draw('e0 x0 same')
                ratios[hstub].Draw('hist same')

        p.adjustLowerHisto(ratios[var[0] + firstname])
        ratios[var[0] + firstname].SetMinimum(1. - var[4] + 0.0001)
        ratios[var[0] + firstname].SetMaximum(1. + var[4] - 0.0001)
        ratios[var[0] + firstname].GetXaxis().SetTitle(var[3])
        ratios[var[0] + firstname].GetYaxis().SetTitle('#scale[0.9]{#frac{FastSim}{FullSim}}')

        if type(var[2]) == list:
            lines[var[0] + padname] = ROOT.TLine(var[2][0], 1., var[2][-1], 1.)
        else:
            lines[var[0] + padname] = ROOT.TLine(var[2][1], 1., var[2][2], 1.)
        lines[var[0] + padname].SetLineWidth(1)
        lines[var[0] + padname].SetLineColor(ROOT.kBlack)
        lines[var[0] + padname].Draw('same')

    canvas[var[0]].Update()
    canvas[var[0]].Draw()
    canvas[var[0]].Print('figs'+training_id+'/'+nameout.replace('VARIABLE', var[0].replace('_', '').replace('CLASS', '')))
