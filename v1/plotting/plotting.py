#! /usr/bin/env python

from ROOT import gROOT, gPad, TPad
import CMS_lumi, tdrstyle

gROOT.SetBatch()        # don't pop up canvases


class Plotting():
    
    def __init__(self, text="13 TeV", extratext="Preliminary", H_ref=600, W_ref=800, iPos=11):

#        self.text = "13 TeV"
#        self.extratext = "Preliminary"
#        self.H_ref = 600
#        self.W_ref = 800
        
        self.text = text
        self.extratext = extratext
        self.H_ref = H_ref
        self.W_ref = W_ref
            
        self.iPos = iPos
        if( self.iPos==0 ): CMS_lumi.relPosX = 0.12
        
        self.iPeriod = 0
        
        self.W = self.W_ref
        self.H  = self.H_ref
        
        self.T = 0.08*self.H_ref
        self.B = 0.12*self.H_ref
        self.L = 0.14*self.W_ref  # 0.12*self.W_ref
        self.R = 0.04*self.W_ref


    def setStyle(self):
    
        tdrstyle.setTDRStyle()
        
        #change the CMS_lumi variables (see CMS_lumi.py)
        CMS_lumi.lumi_7TeV = "4.8 fb^{-1}"
        CMS_lumi.lumi_8TeV = "18.3 fb^{-1}"
        CMS_lumi.writeExtraText = 1
        CMS_lumi.extraText = self.extratext
        CMS_lumi.lumi_sqrtS = self.text # used with iPeriod = 0, e.g. for simulation-only plots (default is an empty string)
    
    
    def preparePad(self):
        
        gPad.SetFillColor(0)
        gPad.SetBorderMode(0)
        gPad.SetFrameFillStyle(0)
        gPad.SetFrameBorderMode(0)
        gPad.SetLeftMargin( self.L/self.W )
        gPad.SetRightMargin( self.R/self.W )
        gPad.SetTopMargin( self.T/self.H )
        gPad.SetBottomMargin( self.B/self.H )
        gPad.SetTickx(1)
        gPad.SetTicky(1)

        
    def postparePad(self):
        
        CMS_lumi.CMS_lumi(gPad, self.iPeriod, self.iPos)
        
        gPad.cd()
        gPad.Update()
        gPad.RedrawAxis()
        frame = gPad.GetFrame()
        frame.Draw()


    def addLowerPads(self, canvas):

        # adjusted from https://root.cern/doc/master/ratioplot_8py.html

        subpads = {}

        for ipad, pad in enumerate(list(canvas.GetListOfPrimitives())):

            ipad += 1

            pad.cd()

            subpads[str(ipad) + '_1'] = TPad(pad.GetName() + '_1', pad.GetName() + '_1', 0., 0.3, 1., 1.0)

            subpads[str(ipad) + '_1'].SetFillColor(0)
            subpads[str(ipad) + '_1'].SetBorderMode(0)
            subpads[str(ipad) + '_1'].SetFrameFillStyle(0)
            subpads[str(ipad) + '_1'].SetFrameBorderMode(0)
            subpads[str(ipad) + '_1'].SetLeftMargin( self.L/self.W )
            subpads[str(ipad) + '_1'].SetRightMargin( self.R/self.W )
            subpads[str(ipad) + '_1'].SetTopMargin( self.T/self.H )
            subpads[str(ipad) + '_1'].SetBottomMargin(0.02)
            subpads[str(ipad) + '_1'].SetTickx(1)
            subpads[str(ipad) + '_1'].SetTicky(1)

            subpads[str(ipad) + '_1'].Draw()

            pad.cd()

            subpads[str(ipad) + '_2'] = TPad(pad.GetName() + '_2', pad.GetName() + '_2', 0., 0., 1., 0.3)

            subpads[str(ipad) + '_2'].SetFillColor(0)
            subpads[str(ipad) + '_2'].SetBorderMode(0)
            subpads[str(ipad) + '_2'].SetFrameFillStyle(0)
            subpads[str(ipad) + '_2'].SetFrameBorderMode(0)
            subpads[str(ipad) + '_2'].SetLeftMargin( self.L/self.W )
            subpads[str(ipad) + '_2'].SetRightMargin( self.R/self.W )
            subpads[str(ipad) + '_2'].SetTopMargin(0)
            # subpads[str(ipad) + '_2'].SetBottomMargin( self.B/self.H )
            subpads[str(ipad) + '_2'].SetBottomMargin( 0.3 )
            subpads[str(ipad) + '_2'].SetTickx(1)
            subpads[str(ipad) + '_2'].SetTicky(1)

            subpads[str(ipad) + '_2'].Draw()


        return subpads

    def adjustLowerHisto(self, h):

        factor = 7. / 3.  # ratio of height of upper pad to lower pad

        x = h.GetXaxis()
        x.SetLabelSize(factor * 0.05)
        x.SetTitleSize(factor * 0.06)

        y = h.GetYaxis()
        y.SetLabelSize(factor * 0.05)
        y.SetTitleSize(factor * 0.06)
        y.SetTitleOffset(0.9 / factor)
        y.CenterTitle()
        y.SetNdivisions(505)

