#! /usr/bin/env python


import ROOT
from plotting import Plotting
from array import array


normalize = True

training_id = '20231105_1'

fname = '/nfs/dust/cms/user/wolfmor/Refinement/TrainingOutput/output_refinement_regression_TRAININGID.root'
fin = ROOT.TFile(fname.replace('TRAININGID', training_id))
tree = fin.Get('tJet')

nameout = 'regression_1D_' + training_id + '_VARIABLE.png'


# (name, suffix, label, color, fillstyle, linewidth, selection, legend option)
samples = [
    ('full', 'FullSim', 'FullSim', ROOT.kGreen+2, 3004, 3, '1&&', 'lf'),
    ('fast', 'FastSim', 'FastSim', ROOT.kRed+1, 0, 3, '1&&', 'l'),
    # ('refined', 'Refined', 'FastSim Refined', ROOT.kAzure+2, 0, 3, '1&&', 'l'),
    ('refined', 'Refined', 'FastSim Refined', ROOT.kAzure+2, 0, 3, 'isTrainValTest<2&&', 'l'),
    ('refinedtest', 'Refined', 'FastSim Refined (Test)', ROOT.kAzure+1, 0, 3, 'isTrainValTest>1&&', 'lp'),
]

# (name, branch name, (nbins, xlow, xhigh), title, ratio plot range, selection)
variables = [
    ('DeepFlavB', 'RecJet_btagDeepFlavB_CLASS', (60, -0.1, 1.1), 'DeepJet b+bb+lepb Discriminator', 0.3, '1')
]

height = 800
width = 800

p = Plotting(
    text='(13 TeV)',
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

        if type(v[2]) == list:
            histos[v[0] + s[0]] = ROOT.TH1F('h' + v[0] + s[0], '', len(v[2])-1, array('d', v[2]))
        else:
            histos[v[0] + s[0]] = ROOT.TH1F('h' + v[0] + s[0], '', v[2][0], v[2][1], v[2][2])

        tree.Draw(v[1].replace('CLASS', s[1]) + '>>h' + v[0] + s[0], s[6] + v[5].replace('CLASS', s[1]), '')

        if normalize: histos[v[0] + s[0]].Scale(1. / histos[v[0] + s[0]].Integral())

        histos[v[0] + s[0]].SetFillStyle(s[4])
        histos[v[0] + s[0]].SetFillColor(s[3])
        histos[v[0] + s[0]].SetLineWidth(s[5])
        if 'test' not in s[0] and 'val' not in s[0]: histos[v[0] + s[0]].SetMarkerSize(0)
        histos[v[0] + s[0]].SetLineColor(s[3])
        histos[v[0] + s[0]].SetMarkerColor(s[3])

    for s in samples:

        if s[0] == 'full': continue

        if not histos[v[0] + s[0]].GetSumw2N(): histos[v[0] + s[0]].Sumw2()
        ratios[v[0] + s[0]] = histos[v[0] + s[0]].Clone('hr' + v[0] + s[0])
        ratios[v[0] + s[0]].UseCurrentStyle()
        ratios[v[0] + s[0]].SetStats(0)
        ratios[v[0] + s[0]].Divide(histos[v[0] + 'full'])
        ratios[v[0] + s[0]].SetFillStyle(s[4])
        ratios[v[0] + s[0]].SetFillColor(s[3])
        ratios[v[0] + s[0]].SetLineWidth(s[5])
        ratios[v[0] + s[0]].SetMarkerSize(0)
        ratios[v[0] + s[0]].SetLineColor(s[3])
        ratios[v[0] + s[0]].SetMarkerColor(s[3])


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
        if 'test' in s[0] or 'val' in s[0]: histos[v[0] + s[0]].Draw('e0 x0 same')
        else: histos[v[0] + s[0]].Draw('hist same')

    globalmin = min([histos[key].GetMinimum(0) for key in histos if v[0] == key[:len(v[0])] and key[len(v[0]):] in [s[0] for s in samples]])
    globalmax = min([histos[key].GetMaximum() for key in histos if v[0] == key[:len(v[0])] and key[len(v[0]):] in [s[0] for s in samples]])
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
            if s[0] == 'full': continue
            if firstname is None: firstname = s[0]
            if 'test' in s[0] or 'val' in s[0]:
                ratios[v[0] + s[0]].Draw('e0 x0 same')
                ratios[v[0] + s[0]].Draw('hist same')
            else:
                ratios[v[0] + s[0]].Draw('e0 x0 same')
                ratios[v[0] + s[0]].Draw('hist same')

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
    canvas[v[0]].Print(nameout.replace('VARIABLE', v[0].replace('_', '').replace('CLASS', '')))
