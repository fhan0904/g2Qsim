#include "TH1.h"
#include "TF1.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TCanvas.h"
#include "TMinuit.h"
#include "TMath.h"

TF1 *fg;
TFile *file0;
TH1D *h;
TH2D *hPedVsTime;
TH1D *hPed = new TH1D( "hPed", "hPed", 35000, -0.5, 34999.5);
TH1D *hPedSig = new TH1D( "hPedSig", "hPedSig", 35000, -0.5, 34999.5);

int dofitped(){

  // read root file, get time distribution 
  file0 = TFile::Open("test.root");
  file0->GetObject( "hFlush2D", hPedVsTime);
  hPedVsTime->SetName("hPedVsTime");
  hPedVsTime->SetTitle("pedestal versus time");
  hPedVsTime->SetLineColor(kBlack);
  
  // pedestal fit function
  double Amp = 100., Ctr = 0.0, Sig = 40.;
  TF1 *fg = new TF1( "fg", "[0]*TMath::Gaus(x,[1],[2])", -1500., 1500.);
  fg->SetParameter( 0, Amp);    
  fg->SetParameter( 1, Ctr);    
  fg->SetParameter( 2, Sig);
  
  int i, bins = hPedVsTime->GetNbinsX(), step = 16;
  double ped, sig;
  for (i = 1; i <= bins; i+=step){ // loop over time bins
    
    // simple pedestal determination from mean of distribution
    // doesnt work well due to presence of pulses 
    // double ped = hHistnoise0->ProjectionY("h1",i,i)->GetMean();
    // double err = hHistnoise0->ProjectionY("h1",i,i)->GetMeanError();
    
    // get single time-bin pedestal histogram (project is ok for different nb's)
    h = hPedVsTime->ProjectionY("h",i,i+step-1);
    
    // iterate fitting to reduce the sensitivity to location of pedestal peak in fit window 
    Amp = 100.;
    Ctr = 0.0; 
    Sig = 40.;
    fg->SetParameter( 0, Amp);    
    fg->SetParameter( 1, Ctr);    
    fg->SetParameter( 2, Sig);
    
    h->Fit( "fg", "QR", "", -100., 100.); // run group 180-269, with average pedestal subtraction 
    ped = fg->GetParameter(1);
    sig = fg->GetParameter(2);
    h->Fit( "fg", "QR", "", ped-2.5*sig, ped+2.5*sig);
    ped = fg->GetParameter(1);
    sig = fg->GetParameter(2);
    h->Fit( "fg", "R", "", ped-2.5*sig, ped+2.5*sig);
    ped = fg->GetParameter(1);
    err = fg->GetParError(1);
    sig = fg->GetParameter(2);
    sigerr = fg->GetParError(2);    
    printf(" i %i, ped %f err %f\n", i, ped, err);
    
    // build histogram of pedestal versus time bin with fit errors
    // (failed to get storage of histogram array to work properly) 
    hPed->SetBinContent( i, ped);
    hPed->SetBinError( i, err);
    hPedSig->SetBinContent( i, sig);
    hPedSig->SetBinError( i, sigerr);
    
  } // loop over time bins
  
}  


