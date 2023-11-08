#! /usr/bin/env python


import ROOT
from plotting import Plotting

rightmargin = 0.19
zaxistitleoffset = 1.1

onlyupperhalf = True
diff = 'divide'  # 'subtract'

training_id = '20231105_1'

fin = ROOT.TFile('/nfs/dust/cms/user/wolfmor/Refinement/TrainingOutput/output_refinement_regression_TRAININGID.root'.replace('TRAININGID', training_id))
tree = fin.Get('tJet')

nameout = 'regression_correlationfactors_' + training_id + '.png'

# (name, suffix, label)
samples = [
    ('full', 'FullSim', 'FullSim'),
    ('fast', 'FastSim', 'FastSim'),
    ('refined', 'Refined', 'FastSim Refined'),
]

# (name/title, branch name, (nbins, xlow, xhigh))
variables = [
    ('GEN p_{T}', 'GenJet_pt', (100, 0., 2000)),
    ('p_{T}', 'RecJet_pt_CLASSNOTREFINED', (100, 0., 1000)),
    ('B', 'RecJet_btagDeepFlavB_CLASS', (120, -0.1, 1.1)),
    ('CvB', 'RecJet_btagDeepFlavCvB_CLASS', (120, -0.1, 1.1)),
    ('CvL', 'RecJet_btagDeepFlavCvL_CLASS', (120, -0.1, 1.1)),
    ('QG', 'RecJet_btagDeepFlavQG_CLASS', (120, -0.1, 1.1)),
]
numvars = len(variables)

height = 600
width = 800

ndigits = 2
whitethreshold = 0.8

p = Plotting(
    text='',
    extratext='Simulation',
    H_ref=height,
    W_ref=width,
    iPos=0
)

p.setStyle()

c0 = ROOT.TCanvas('c0', 'c0', 1, 1)

histos = {}
corrhistos = {}
for s in samples:

    print(s[0])

    histos[s[0]] = ROOT.TH2D('hCorrFactors' + s[0], '', numvars, 0, numvars, numvars, 0, numvars)
    histos[s[0] + 'diffFull'] = ROOT.TH2D('hCorrFactorsDiffFull' + s[0], '', numvars, 0, numvars, numvars, 0, numvars)

    histos[s[0] + 'white'] = ROOT.TH2D('hCorrFactorsWhite' + s[0], '', numvars, 0, numvars, numvars, 0, numvars)
    histos[s[0] + 'diffFull' + 'white'] = ROOT.TH2D('hCorrFactorsDiffFullWhite' + s[0], '', numvars, 0, numvars, numvars, 0, numvars)

    for ix, x in enumerate(variables):

        histos[s[0]].GetXaxis().SetBinLabel(ix+1, x[0])
        histos[s[0] + 'diffFull'].GetXaxis().SetBinLabel(ix+1, x[0])

        for iy, y in enumerate(variables):

            if ix == 0:
                histos[s[0]].GetYaxis().SetBinLabel(iy+1, y[0])
                histos[s[0] + 'diffFull'].GetYaxis().SetBinLabel(iy+1, y[0])

            if onlyupperhalf and iy <= ix: continue

            corrhistos[s[0] + x[0] + y[0]] = ROOT.TH2F('h' + s[0] + x[0] + y[0], '', x[2][0], x[2][1], x[2][2], y[2][0], y[2][1], y[2][2])

            tree.Draw(y[1].replace('CLASS', s[1]).replace('RefinedNOTREFINED', 'FastSimNOTREFINED').replace('NOTREFINED', '')
                      + ':'
                      + x[1].replace('CLASS', s[1]).replace('RefinedNOTREFINED', 'FastSimNOTREFINED').replace('NOTREFINED', '')
                      + '>>h' + s[0] + x[0] + y[0], '1', '')

            corrfactor = corrhistos[s[0] + x[0] + y[0]].GetCorrelationFactor()

            histos[s[0]].SetBinContent(ix+1, iy+1, round(corrfactor, ndigits))
            if abs(corrfactor) > whitethreshold:
                histos[s[0] + 'white'].SetBinContent(ix+1, iy+1, round(corrfactor, ndigits))

            if diff == 'subtract':
                histos[s[0] + 'diffFull'].SetBinContent(ix+1, iy+1, abs(round(corrfactor, ndigits) - round(corrhistos['full' + x[0] + y[0]].GetCorrelationFactor(), ndigits)))
                if abs(abs(round(corrfactor, ndigits) - round(corrhistos['full' + x[0] + y[0]].GetCorrelationFactor(), ndigits))) > whitethreshold:
                    histos[s[0] + 'diffFull' + 'white'].SetBinContent(ix+1, iy+1, abs(round(corrfactor, ndigits) - round(corrhistos['full' + x[0] + y[0]].GetCorrelationFactor(), ndigits)))
            elif diff == 'divide':
                if not round(corrhistos['full' + x[0] + y[0]].GetCorrelationFactor(), ndigits) == 0:
                    histos[s[0] + 'diffFull'].SetBinContent(ix+1, iy+1, 1 - round(corrfactor, ndigits) / round(corrhistos['full' + x[0] + y[0]].GetCorrelationFactor(), ndigits))
                    if abs(1 - round(corrfactor, ndigits) / round(corrhistos['full' + x[0] + y[0]].GetCorrelationFactor(), ndigits)) > whitethreshold:
                        histos[s[0] + 'diffFull' + 'white'].SetBinContent(ix+1, iy+1, 1 - round(corrfactor, ndigits) / round(corrhistos['full' + x[0] + y[0]].GetCorrelationFactor(), ndigits))
                else:
                    if round(corrfactor, ndigits) == 0:
                        histos[s[0] + 'diffFull'].SetBinContent(ix+1, iy+1, 0.)
                    else:
                        histos[s[0] + 'diffFull'].SetBinContent(ix+1, iy+1, 1.)
                        histos[s[0] + 'diffFull' + 'white'].SetBinContent(ix+1, iy+1, 1.)
            else:
                raise NotImplementedError('what\'s diff?')

canvas = ROOT.TCanvas('c', 'c', 3*width, 2*height)
canvas.Divide(len(samples), 2)

for ipad, s in enumerate(samples):

    canvas.cd(ipad+1)

    p = Plotting(
        text='(13 TeV)',
        extratext='  Simulation',
        H_ref=height,
        W_ref=width,
        iPos=0
    )

    p.setStyle()

    ROOT.gStyle.SetPalette(ROOT.kGreenPink)
    ROOT.gStyle.SetNumberContours(101)
    ROOT.gStyle.SetPaintTextFormat('4.' + str(ndigits) + 'f')

    p.preparePad()

    ROOT.gPad.SetRightMargin(rightmargin)
    histos[s[0]].GetZaxis().SetTitle('r_{xy}(' + s[2] + ')')
    histos[s[0]].GetZaxis().SetTitleOffset(zaxistitleoffset)
    histos[s[0]].GetZaxis().SetLabelSize(histos[s[0]].GetXaxis().GetLabelSize())
    histos[s[0]].GetZaxis().SetRangeUser(-1., 1.)
    histos[s[0]].SetMarkerSize(2.)

    histos[s[0]].Draw('text colz')

    histos[s[0] + 'white'].SetMarkerSize(2.)
    histos[s[0] + 'white'].SetMarkerColor(ROOT.kWhite)
    histos[s[0] + 'white'].Draw('text same')

    p.postparePad()

for ipad, s in enumerate(samples):

    canvas.cd(ipad+1+len(samples))

    p = Plotting(
        text='(13 TeV)',
        extratext='  Simulation',
        H_ref=height,
        W_ref=width,
        iPos=0
    )

    p.setStyle()

    ROOT.gStyle.SetPalette(ROOT.kGreenPink)
    ROOT.gStyle.SetNumberContours(101)
    ROOT.gStyle.SetPaintTextFormat('4.' + str(ndigits) + 'f')

    p.preparePad()

    ROOT.gPad.SetRightMargin(rightmargin)
    if diff == 'divide':
        histos[s[0] + 'diffFull'].GetZaxis().SetTitle('1 - r_{xy}(' + s[2] + ') / r_{xy}(FullSim)')
    elif diff == 'subtract':
        histos[s[0] + 'diffFull'].GetZaxis().SetTitle('|r_{xy}(' + s[2] + ') - r_{xy}(FullSim)|')
    histos[s[0] + 'diffFull'].GetZaxis().SetTitleOffset(zaxistitleoffset)
    histos[s[0] + 'diffFull'].GetZaxis().SetLabelSize(histos[s[0]].GetXaxis().GetLabelSize())
    histos[s[0] + 'diffFull'].GetZaxis().SetRangeUser(-1., 1.)
    histos[s[0] + 'diffFull'].SetMarkerSize(2.)

    histos[s[0] + 'diffFull'].Draw('text colz')

    histos[s[0] + 'diffFull' + 'white'].SetMarkerSize(2.)
    histos[s[0] + 'diffFull' + 'white'].SetMarkerColor(ROOT.kWhite)
    histos[s[0] + 'diffFull' + 'white'].Draw('text same')

    p.postparePad()

canvas.Update()
canvas.Draw()
canvas.Print(nameout)
