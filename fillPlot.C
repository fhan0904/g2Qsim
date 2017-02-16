TH1F *hFill;

void fillPlot(){

  ifstream infile;
  Int_t bin; 
  float energy;
  hFill = new TH1F("hFill", "hFill", 524288/32, 0.0, 524288/32. );
  hFill->SetLineWidth(3.0);
  hFill->SetLineColor(kRed);
  hFill->SetTitle("fill energy distribution");
  hFill->SetXTitle("clock ticks [1ns/tick]");
  hFill->SetYTitle("energy [MeV]");

  infile.open("fillSumArray.dat");

  while ( !infile.eof() ) {
    infile >> bin >> energy;
    if (energy >= 1.e9) energy = 0.0;
    if (bin >= 524244/32.-1) break;
    hFill->Fill( bin, energy);
    printf("bin %i, energy %f\n", bin, energy);
  }

  //hFill->Rebin(64);
  hFill->Draw();
  return;
}
