#include "TH1.h"
#include "TF1.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TCanvas.h"
#include "TMinuit.h"
#include "TMath.h"

TFile *file0;
TF1 *precessf, *sigmaf;
TH1D *hTime, *hResi;
TCanvas *ct = new TCanvas("ct","time distribution",1400,700);

Double_t fprec(Double_t *x, Double_t *par)
{
  // simple precession function with sinosoidal amplitude, freqeuncy and phase
  //
  // par[0] - N
  // par[1] - lambda
  // par[2] - A
  // par[3] - omega_a
  // par[4] - phi
  // par[5] - bkd
     
  Double_t xx =x[0];
  Double_t f = par[0] * exp( -xx * par[1] ) * ( 1.0 + par[2]*cos( par[3]*xx + par[4] ) ) + par[5];
     
  return f;
}

Double_t fsigma(Double_t *x, Double_t *par)
{
  // simple sigma function with counting statistics and uniform noise term
  //
  // par[0] - sigma1^2 for counting statistics
  // par[1] - sigma2^2 for uniform noise
     
  Double_t xx =x[0];
  if (xx >= 12000. && xx <= 12100.) printf("xx %f, precessf->Eval( hTime->GetBinCenter(xx) %f, par[0] %f, par[1] %f\n", xx, precessf->Eval( hTime->GetBinCenter(xx) ), par[0], par[1]);
  Double_t f = par[0] * precessf->Eval( hTime->GetBinCenter(xx) ) + par[1];

  return f;
}

void dofit(){

  // read root file, get time distribution 
  file0 = TFile::Open("test.root");
  file0->GetObject( "hFlush1D", hTime);
  hTime->SetName("hTime");
  hTime->SetTitle("positron time distribution");
  hTime->SetLineColor(kBlack);

  // Qmethod rebin factor
  double nsToBin = 16.;

  // function range
  double minT = 0.0/nsToBin, maxT = 700000./nsToBin;

  // precession function parameters
  // in rand.cu
  double tau0 = 6.4e4; //ns
  double omega_a0 = 1.438e-3; // rad/ns

  //  fit initial values
  double tau = 64100.; // time-dilated muon lifetime [ns]
  double omega_a = 1.4381e-3; // anomalous precession frequency [ns]
  omega_a = omega_a*nsToBin; // convert to Q-method bins
  double lambda = (1./tau)*nsToBin; // convert to rate and Q-method bins
  double N0 = 3.5e8 , ampl = 0.4,  phase = TMath::Pi(), bkgd = 0.0;


  // precession function definition
  precessf = new TF1( "precessf", fprec, minT, maxT, 6);
  precessf->SetParNames("N0","lambda","amplitude","omega_a","phase","bkgd");
  precessf->SetParameters( N0, lambda, ampl, omega_a, phase, bkgd);
  precessf->SetLineColor(kRed);
  precessf->SetLineWidth(2);
  precessf->SetNpx(10000);

  // statistics parameters
  double sig12 = 1.0, sig22 = 0.0;

  // sigma^2 function definition
  sigmaf = new TF1( "sigmaf", fsigma, minT, maxT, 2);
  sigmaf->SetParNames("sigma1^2","sigma2^2");
  sigmaf->SetParameters( sig12, sig22);
  sigmaf->SetLineColor(kRed);
  sigmaf->SetLineWidth(2);
  sigmaf->SetNpx(10000);

  // fit range 
  double minTF = 10000.0/nsToBin, maxTF = 400000./nsToBin;
  //precessf->FixParameter( 1, 0.0);  
  //precessf->FixParameter( 1, 1./lambda);  
  //precessf->FixParameter( 2, ampl);  
  //precessf->FixParameter( 3, omega_a);
  //precessf->FixParameter( 4, phase);
  printf("start fit ...\n");
  hTime->Fit(precessf,"","RV", minTF, maxTF);  
  hTime->Draw();

  double tauF = 1./(precessf->GetParameter(1)/nsToBin);  
  double dtauF = tauF*(precessf->GetParError(1)/precessf->GetParameter(1));  
  double omega_aF = precessf->GetParameter(3)/nsToBin;  
  double domega_aF = omega_aF*precessf->GetParError(3)/precessf->GetParameter(3);  
  double DeltatauF = (tauF - tau0)/tau0;
  double Deltaomega_aF = (omega_aF - omega_a0)/omega_a0;
  double SigDeltauF = (tauF - tau0)/dtauF;
  double SigDelomega_aF = (omega_aF - omega_a0)/domega_aF;

  double ndf = precessf->GetNDF();
  double chiSq = precessf->GetChisquare();

  printf(" tau = %e +/- %e ns, omega_a =  %e +/- %e ns\n", tauF, dtauF, omega_aF, domega_aF);
  printf(" NDF %i, Chi-squared %f, Delta tau = %e (%06f2.1 sig), Delta omega_a =  %e (%06f2.1 sig)\n", ndf, chiSq, DeltatauF, SigDeltauF, Deltaomega_aF, SigDelomega_aF);

  for (int ib =  minTF; ib <= maxTF; ib++){
    hTime->SetBinError(  ib, sqrt(hTime->GetBinContent(ib)*chiSq/ndf) ); 
  }
  hTime->Fit(precessf,"","RV", minTF, maxTF);  
  hTime->Draw();

  tauF = 1./(precessf->GetParameter(1)/nsToBin);  
  dtauF = tauF*(precessf->GetParError(1)/precessf->GetParameter(1));  
  omega_aF = precessf->GetParameter(3)/nsToBin;  
  domega_aF = omega_aF*precessf->GetParError(3)/precessf->GetParameter(3);  
  DeltatauF = (tauF - tau0)/tau0;
  Deltaomega_aF = (omega_aF - omega_a0)/omega_a0;
  SigDeltauF = (tauF - tau0)/dtauF;
  SigDelomega_aF = (omega_aF - omega_a0)/domega_aF;
  
  ndf = precessf->GetNDF();
  chiSq = precessf->GetChisquare();
  
  printf(" tau = %e +/- %e ns, omega_a =  %e +/- %e ns\n", tauF, dtauF, omega_aF, domega_aF);
  printf(" NDF %i, Chi-squared %f, Delta tau = %e (%06f2.1 sig), Delta omega_a =  %e (%06f2.1 sig)\n", ndf, chiSq, DeltatauF, SigDeltauF, Deltaomega_aF, SigDelomega_aF);

  hResi=(TH1D*)hTime->Clone();
  hResi->Reset();
  hResi->SetName("hResi");
  hResi->SetTitle("(data - fit) residuals time distribution");
  hResi->SetLineColor(kBlue);
  hResi2=(TH1D*)hTime->Clone();
  hResi2->Reset();
  hResi2->SetName("hResi2");
  hResi2->SetTitle("(data - fit)^2 residuals time distribution");
  hResi2->SetLineColor(kBlue);
  for (int ib =  minTF; ib <= maxTF; ib++){
    double resi = ( hTime->GetBinContent(ib) - precessf->Eval( hTime->GetBinCenter(ib) ) ); 
    hResi->SetBinContent( ib, resi );
    hResi2->SetBinContent( ib, resi*resi );
   //printf("ib %i, resi %f\n", ib, resi);
  }

  /*
  sig12 = hResi2->Integral(minTF,maxTF)/hTime->Integral(minTF,maxTF);
  sigmaf->SetParameters( sig12, sig22);
  sigmaf->FixParameter( 1, 0.0);
  hResi2->Draw();
  sigmaf->Draw("same");
  hResi2->Fit(sigmaf,"","RW", minTF, maxTF);  
  hResi2->Draw();
  */
}
