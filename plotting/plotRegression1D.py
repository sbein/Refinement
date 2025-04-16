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


# (name, suffix, label, color, fillstyle, linewidth, selection, legend option)
samples = [
    ('full', 'FullSim', 'FullSim', ROOT.kGreen+2, 3004, 3, '1&&', 'lf'),
    ('fast', 'FastSim', 'FastSim', ROOT.kRed+1, 0, 3, '1&&', 'l'),
    # ('refined', 'Refined', 'FastSim Refined', ROOT.kAzure+2, 0, 3, '1&&', 'l'),
    ('refined', 'Refined', 'FastSim Refined', ROOT.kAzure+2, 0, 3, 'isTrainValTest<2&&', 'l'),
    ('refinedtest', 'Refined', 'FastSim Refined (Test)', ROOT.kAzure+1, 0, 3, 'isTrainValTest>1&&', 'lp'),
]

# (name, branch name, (nbins, xlow, xhigh), title, ratio plot range, selection)
##variables = [
##    ('DeepFlavB', 'RecJet_btagDeepFlavB_CLASS', (60, -0.1, 1.1), 'DeepJet b+bb+lepb Discriminator', 0.3, '1')
##]
nbins = 60
#nbins = 20
variables = [
    ('JetPt',     'RecJet_pt_CLASS', (nbins, 0, 1000),'Jet p_{T} [GeV]', 0.3, '1'),
    ('DeepFlavB', 'RecJet_btagDeepFlavB_CLASS', (nbins, -0.1, 1.1),'DeepJet b+bb+lepb Discriminator', 0.3, '1'),
    ('DeepFlavCvB', 'RecJet_btagDeepFlavCvB_CLASS', (nbins, -0.1, 1.1),'DeepJet C/B Discriminator', 0.3, '1'),
    ('DeepFlavCvL', 'RecJet_btagDeepFlavCvL_CLASS', (nbins, -0.1, 1.1),'DeepJet C/L Discriminator', 0.3, '1'),
    ('DeepFlavQG', 'RecJet_btagDeepFlavQG_CLASS', (nbins, -0.1, 1.1),'DeepJet Q/g Discriminator', 0.3, '1'),
    ('UParTAK4B', 'RecJet_btagUParTAK4B_CLASS', (nbins, -0.1, 1.1),'UParTAK4 B Discriminator', 0.3, '1'),
    ('UParTAK4CvB', 'RecJet_btagUParTAK4CvB_CLASS', (nbins, -0.1, 1.1),'UParTAK4 C/B Discriminator', 0.3, '1'),
    ('UParTAK4CvL', 'RecJet_btagUParTAK4CvL_CLASS', (nbins, -0.1, 1.1),'UParTAK4 C/L Discriminator', 0.3, '1'),
    ('UParTAK4QG', 'RecJet_btagUParTAK4QvG_CLASS', (nbins, -0.1, 1.1),'UParTAK4 Q/g Discriminator', 0.3, '1'),
    
]
if 'Classifier' in training_id: variables.append(('Z classifier', 'RecJet_ffClassifier_CLASS', (nbins, -0.1, 1.1),'Z(Fast-Full classifier)', 0.3, '1'))

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
for v in variables:

    print(v[0])

    c0.cd()

    for s in samples:
        hstub = v[0] + s[0]
        if type(v[2]) == list:
            histos[hstub] = ROOT.TH1F('h' + hstub, '', len(v[2])-1, array('d', v[2]))
        else:
            histos[hstub] = ROOT.TH1F('h' + hstub, '', v[2][0], v[2][1], v[2][2])

        tree.Draw(v[1].replace('CLASS', s[1]) + '>>h' + hstub, s[6] + v[5].replace('CLASS', s[1]), '')

        if normalize: histos[hstub].Scale(1. / histos[hstub].Integral())

        histos[hstub].SetFillStyle(s[4])
        histos[hstub].SetFillColor(s[3])
        histos[hstub].SetLineWidth(s[5])
        if 'test' not in s[0] and 'val' not in s[0]: histos[hstub].SetMarkerSize(0)
        histos[hstub].SetLineColor(s[3])
        histos[hstub].SetMarkerColor(s[3])

    for s in samples:

        if s[0] == 'full': continue
        hstub = v[0] + s[0]
        if not histos[hstub].GetSumw2N(): histos[hstub].Sumw2()
        ratios[hstub] = histos[hstub].Clone('hr' + hstub)
        ratios[hstub].UseCurrentStyle()
        ratios[hstub].SetStats(0)
        ratios[hstub].Divide(histos[v[0] + 'full'])
        ratios[hstub].SetFillStyle(s[4])
        ratios[hstub].SetFillColor(s[3])
        ratios[hstub].SetLineWidth(s[5])
        ratios[hstub].SetMarkerSize(0)
        ratios[hstub].SetLineColor(s[3])
        ratios[hstub].SetMarkerColor(s[3])


    leg[v[0]] = ROOT.TLegend(0.5, 0.6, 0.95, 0.9)

    canvas[v[0]] = ROOT.TCanvas('c' + v[0], 'c' + v[0], 2*width, height)
    canvas[v[0]].Divide(2, 1)

    subpads[v[0]] = p.addLowerPads(canvas[v[0]])

    subpads[v[0]]['1_1'].cd()

    for s in samples:
        if 'test' in s[0] or 'val' in s[0]: histos[v[0] + s[0]].Draw('e0 x0 same')
        else: histos[v[0] + s[0]].Draw('hist same')
        leg[v[0]].AddEntry(histos[v[0] + s[0]], s[2], s[7] if len(s) > 7 else 'lpf')

    histos[v[0] + samples[0][0]].SetMaximum(1.6*max([histos[key].GetMaximum() for key in histos if v[0] == key[:len(v[0])] and key[len(v[0]):] in [s[0] for s in samples]]))
    histos[v[0] + samples[0][0]].GetXaxis().SetTitle(v[3])
    histos[v[0] + samples[0][0]].GetXaxis().SetLabelSize(0)
    if normalize: histos[v[0] + samples[0][0]].GetYaxis().SetTitle('Fraction of Jets')
    else: histos[v[0] + samples[0][0]].GetYaxis().SetTitle('Jets')

    leg[v[0]].Draw('same')

    p.postparePad()

    subpads[v[0]]['2_1'].cd()

    histos[v[0] + 'emptyloghist'] = histos[v[0] + samples[0][0]].Clone('emptyloghist')
    histos[v[0] + 'emptyloghist'].Reset()
    histos[v[0] + 'emptyloghist'].Draw('AXIS')

    for s in samples:
        hstub = v[0] + s[0]
        if 'test' in s[0] or 'val' in s[0]: histos[hstub].Draw('e0 x0 same')
        else: histos[hstub].Draw('hist same')

    #globalmin = min([histos[key].GetMinimum(0) for key in histos if v[0] == key[:len(v[0])] and key[len(v[0]):] in [s[0] for s in samples]])
    #globalmax = min([histos[key].GetMaximum() for key in histos if v[0] == key[:len(v[0])] and key[len(v[0]):] in [s[0] for s in samples]])
    
    globalmin = histos[v[0] + samples[0][0]].GetMinimum(0)
    globalmax = histos[v[0] + samples[0][0]].GetMaximum()
        
    
    logrange = ROOT.TMath.Log10(globalmax) - ROOT.TMath.Log10(globalmin)

    histos[v[0] + 'emptyloghist'].SetMinimum(0.5 * globalmin)
    histos[v[0] + 'emptyloghist'].SetMaximum(globalmax * 10 ** max(1, logrange))
    histos[v[0] + 'emptyloghist'].GetXaxis().SetTitle(v[3])
    if normalize: histos[v[0] + 'emptyloghist'].GetYaxis().SetTitle('Fraction of Jets')
    else: histos[v[0] + 'emptyloghist'].GetYaxis().SetTitle('Jets')

    leg[v[0]].Draw('SAME')

    ROOT.gPad.SetLogy()

    p.postparePad()

    for padname in ['1_2', '2_2']:

        subpads[v[0]][padname].cd()

        firstname = None
        for s in samples:
            hstub = v[0] + s[0]
            if s[0] == 'full': continue
            if firstname is None: firstname = s[0]
            if 'test' in s[0] or 'val' in s[0]:
                ratios[hstub].Draw('e0 x0 same')
                ratios[hstub].Draw('hist same')
            else:
                ratios[hstub].Draw('e0 x0 same')
                ratios[hstub].Draw('hist same')

        p.adjustLowerHisto(ratios[v[0] + firstname])
        ratios[v[0] + firstname].SetMinimum(1. - v[4] + 0.0001)
        ratios[v[0] + firstname].SetMaximum(1. + v[4] - 0.0001)
        ratios[v[0] + firstname].GetXaxis().SetTitle(v[3])
        ratios[v[0] + firstname].GetYaxis().SetTitle('#scale[0.9]{#frac{FastSim}{FullSim}}')

        if type(v[2]) == list:
            lines[v[0] + padname] = ROOT.TLine(v[2][0], 1., v[2][-1], 1.)
        else:
            lines[v[0] + padname] = ROOT.TLine(v[2][1], 1., v[2][2], 1.)
        lines[v[0] + padname].SetLineWidth(1)
        lines[v[0] + padname].SetLineColor(ROOT.kBlack)
        lines[v[0] + padname].Draw('same')

    canvas[v[0]].Update()
    canvas[v[0]].Draw()
    canvas[v[0]].Print('figs'+training_id+'/'+nameout.replace('VARIABLE', v[0].replace('_', '').replace('CLASS', '')))
