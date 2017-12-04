#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <vector>
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <algorithm>
#include <sstream>
#include <string>
#include <Eigen/Dense>

#define OPTIMIZE_DIPOLE_DIPOLE
// #define OPTIMIZE_EXCHANGE

#include "odmr_triplets.cc"
#include "genetic.cc"

class ODMR_Matrix { 
    double Bmin, Bmax, FreqMin, FreqMax, dB, dFreq;
    int NX, NY;
    double dx, dy;
public : 
    std::string source_filename;      
    Eigen::MatrixXf odmr;    
    enum SignalType { SIGNAL_X, SIGNAL_Y, SIGNAL_AMP };


    const double B_scale;
    const double freq_scale;
    const double amp_scale;
    double Xmax, Ymax, Xmin, Ymin;
   
    ODMR_Matrix(void) : B_scale(2.8024e4), freq_scale(1e-6) , amp_scale(1e4) {
       // magnetic field and frequency to MHz
    }

    int Xsize(void) { return NX; }
    int Ysize(void) { return NY; }

    void load_parameters(void) {    
       std::ifstream infile(source_filename.c_str());    
       std::string line;

       bool first = true;
       double prevFreq;
       double prevB;
       
       while (std::getline(infile, line)) {  
	 std::istringstream iss(line);
	 double Freq,FreqBis,B;
	 if ((iss >> Freq >> FreqBis >> B)) { 
	   
	   if (first) { 
	     Bmin = Bmax = B;
	     FreqMin = FreqMax = Freq;
	   } 
	   
	   if (Freq < FreqMin) FreqMin = Freq;
	   if (Freq > FreqMax) FreqMax = Freq;
	   if (B < Bmin) Bmin = B;
	   if (B > Bmax) Bmax = B;
	   
	   if (!first && Freq > prevFreq) {
	     dFreq = Freq - prevFreq;
	   } 
	   if (!first && B > prevB) {
	     dB = B - prevB;
	   } 
	   
	   prevFreq = Freq;
	   prevB = B;	   
	   first = false;
	 } 
       }

       dx = dFreq * freq_scale;
       dy = dB * B_scale;

       Xmin = FreqMin * freq_scale;
       Xmax = FreqMax * freq_scale;
       Ymin = Bmin * B_scale;
       Ymax = Bmax * B_scale;
    }

    int enc_x(double x) { 
        return floor( (x - Xmin) / dx );
    }

    int enc_y(double y) { 
        return floor( (y - Ymin) / dy );
    }

    int dec_x(int x) { 
        return (double) x * dx + Xmin;
    }

    int dec_y(int y) { 
        return (double) y * dy + Ymin;
    }


    void load_matrix(SignalType sign_type = SIGNAL_AMP) { 
       NY = ceil( (Ymax - Ymin) / dy);
       NX = ceil( (Xmax - Xmin) / dx);
       //       std::cerr << NX << std::endl;
       //       std::cerr << NY << std::endl;
       odmr = Eigen::MatrixXf::Zero(NX, NY);

       std::ifstream infile(source_filename.c_str());    
       std::string line;

       std::cout << "# load_matrix signal type " << sign_type << std::endl;
       while (std::getline(infile, line)) {  
	 std::istringstream iss(line);
	 double Freq,FreqBis,B,Icoil,vx,vy;
	 if ((iss >> Freq >> FreqBis >> B >> Icoil >> vx >> vy)) { 
	    double x = Freq * freq_scale;
	    double y = B * B_scale;
	    if (x >= Xmin && x < Xmax && y >= Ymin && y < Ymax) { 
	       switch (sign_type) { 
	       case SIGNAL_X: 
		  odmr( enc_x(x), enc_y(y) ) = amp_scale * vx;	      
		  break;
	       case SIGNAL_Y: 
		  odmr( enc_x(x), enc_y(y) ) = amp_scale * vy;	      
		  break;
	       case SIGNAL_AMP: 
		  odmr( enc_x(x), enc_y(y) ) = amp_scale * sqrt(vx*vx+vy*vy);
		  break;
	       }
	    }
	 }
       }
    }

    void print_matrix(void) { 
       for (int j = 0; j < NY; j++) { 
	  for (int i = 0; i < NX; i++) { 
	    std::cout << dec_x(i) << "     " << dec_y(j) << "     " << odmr(i ,j) << std::endl;
	  }
	  std::cout << std::endl;
       }
    }

    void print_info(void) { 
       std::cout << "# Bmin, Bmax, FreqMin, FreqMax, dB, dFreq " << std::endl;
       std::cout << "# " << Bmin << "  " << Bmax << "  " << FreqMin << "  " << FreqMax << "  " << dB << "  " << dFreq << std::endl;
       std::cout << "# NX, NY " << std::endl;
       std::cout << "# " << NX << "    " << NY << std::endl;
       std::cout << "# dx, dy " << std::endl;
       std::cout << "# " << dx << "    " << dy << std::endl;
       std::cout << "# Bscale, freq_scale, amp_scale " << std::endl;
       std::cout << "# " << B_scale << "   " << freq_scale << "   " << amp_scale << std::endl;
       std::cout << "# Xmax, Ymax, Xmin, Ymin " << std::endl;
       std::cout << "# " << Xmax << "  " << Ymax << "  " << Xmin << "  " << Ymin  << std::endl;
    }
};


template <class Spin> class Spin_From_Gene {
public: 
   enum TPenum { D, E, PHI, UZ, THETA, GAMMA, AMP, NTP } ;

  // optimize population
   enum { NPARAMETERS = NTP + Spin::matrix_size };
  // don't optimize population 
  //   enum { NPARAMETERS = NTP };

   bool inline optimize_population(void) { 
       return (NPARAMETERS - NTP == Spin::matrix_size);
   }

   enum { NSPINS = 1 };
   enum { NVARS = NSPINS * NPARAMETERS };

   std::vector<Spin> spins;
   std::vector<double> Amps;
   std::vector<double> signal;
   std::vector<double> rho0;

   ODMR_Matrix data;
   
   double Dmax;
   double Emax;
   double Tmax;
   double Amp_max;
   double Gamma_max;

   Spin_From_Gene<Spin>(void) : spins(NSPINS), Amps(NSPINS), rho0(Spin::matrix_size) { 
   }

   void triplet_from_vector(int index, const std::vector<double> &pairvec, double Bz)
   {
       spins[index].S.D = pairvec[D];
       spins[index].S.E = pairvec[E];
       spins[index].S.rot.angle_phi_uz(pairvec[THETA], pairvec[PHI], pairvec[UZ]);
       spins[index].S.B << 0, 0, Bz;
       spins[index].update_hamiltonian();
       spins[index].diag(); 
   }

   void print_triplet_parameters(void) { 
      for (int i = 0; i < NSPINS; i++) { 
	 std::cout << "# D " << spins[i].S.D << std::endl;
	 std::cout << "# E " << spins[i].S.E << std::endl;
	 std::cout << "# B " << spins[i].S.B << std::endl;
	 std::cout << "# M " << spins[i].S.rot.matrix()  << std::endl;
      }
  }

   double upper(int i) { 
      int modi = i % NPARAMETERS;
      if (modi < NTP) { 
	 TPenum index = static_cast<TPenum>(modi);
	 switch (index) {
         case D: 
	    return Dmax;
         case E:
	    return Emax;
         case UZ: 
	    return 1.0;
         case PHI: case THETA: 
	    return M_PI;
         case AMP:
  	    return Amp_max;
         case GAMMA:
	    return Gamma_max;
	 }
      } else  {
	 return 1.0;
      }
   }

   double lower(int i) { 
      int modi = i % NPARAMETERS;
      if (modi < NTP) { 
	 TPenum index = static_cast<TPenum>(modi);
	 switch (index) {
         case GAMMA: 
	   return 0.0; 
         case AMP: 
	   return 0.0; 
         default : 
	   return -upper(i);
	 }
      } else  {
	 return 0.0;
      }
   }

   void update_spins_from_gene_at_Bz(const double gene[], double Bz) { 
      std::vector<double> pairvec(NPARAMETERS);
      for (int i = 0; i < NSPINS; i++) { 
	 for (int j = 0; j < NPARAMETERS; j++) { 
	    pairvec[j] = gene[i * NPARAMETERS + j];
	 }
	 triplet_from_vector(i, pairvec, Bz);
	 Amps[i] = pairvec[AMP];
      }
   }

   double gamma_from_gene(int pair_number, const double gene[]) { 
       return gene[pair_number * NPARAMETERS + GAMMA];
   }

   void pop_from_gene(int pair_number, const double gene[]) { 
       for (int i = NTP; i < NPARAMETERS; i++) { 
	  rho0[i - NTP] = gene[pair_number * NPARAMETERS + i];
       }
   }

   void comp_signal_for_omega(const double gene[]) {
      int NX = data.Xsize();
      signal.resize(data.Xsize());
      for (int i = 0; i < NX; i++) signal[i] = 0.0;

      for (int n = 0; n < NSPINS; n++) { 
	 ESR_Signal<Spin> esr_from_spins(spins[n]);    
	 esr_from_spins.update_from_spin_hamiltonian();
	 esr_from_spins.gamma = gamma_from_gene(n, gene);
	 esr_from_spins.gamma_diag = gamma_from_gene(n, gene);

	 if (optimize_population()) { 
	    pop_from_gene(n, gene);
	    esr_from_spins.load_rho0(rho0);
	 } else { 
	    esr_from_spins.load_rho0_thermal(Tmax);
	 }

	 for (int i = 0; i < NX; i++) { 
	   double omega = data.dec_x(i);
	   signal[i] += Amps[n] * imag(esr_from_spins.chi1(omega));
	 }
      }
   }

   virtual double score(const double gene[]) { 
      int NB = data.Ysize();
      int Nomega = data.Xsize();

      double gene_score = 0.0;
      for (int b = 0; b < NB; b++) { 
	 double Bz = data.dec_y(b);
	 update_spins_from_gene_at_Bz(gene, Bz);
	 comp_signal_for_omega(gene);
	 for (int w = 0; w < Nomega; w++) { 
	    gene_score += fabs( data.odmr(w, b) - signal[w] );
	 }
      }
      //      print_triplet_parameters();
      return 1.0 / gene_score;
   }

    void read_gene(const char *filename, double gene[], int nparameters) {  
       std::ifstream infile(filename);    
       std::string line;

       int i = 0;
       while (std::getline(infile, line)) {  
	 std::istringstream iss(line);
	 double niter,index,val;
	 if ((iss >> niter >> index >> val)) { 
	    gene[i] = val;
	 }
	 i++;
       }
       if (i != nparameters) { 
	  std::cerr << "read_gene - incorrect read " << i << "  " << nparameters << std::endl;
       }
   }

   void read_gene(const char *filename, double gene[]) { 
      read_gene(filename, gene, NPARAMETERS * NSPINS); 
   }

   void print_gene(const double gene[]) { 
      int NB = data.Ysize();
      int Nomega = data.Xsize();

      for (int b = 0; b < NB; b++) { 
	 double Bz = data.dec_y(b);	
	 update_spins_from_gene_at_Bz(gene, Bz);
	 if (!b) { 
	    print_triplet_parameters();      
	    std::cout << "# printing gene with population " << std::endl;
	    for (int n = 0; n < NSPINS; n++) { 
	       for (int index = 0; index < NPARAMETERS; index++) { 
		  std::cout << "#" << 0 << "   " << index << "   " << gene[n * NPARAMETERS + index] << std::endl;
	       }
	    }
	 }

	 comp_signal_for_omega(gene);
	 for (int w = 0; w < Nomega; w++) { 
	    double omega = data.dec_x(w);
	    std::cout << omega << "   " << Bz << "   " <<  data.odmr(w, b) << "   " << signal[w] << std::endl;
	 }
	 std::cout << std::endl;
      }
   }

   void print_info(void) { 
     std::cout << "# spin_matrix_size " << Spin::matrix_size << std::endl;
     std::cout << "# Dmax " <<  Dmax << std::endl;
     std::cout << "# Emax " <<  Emax << std::endl;
     std::cout << "# Amp_max " <<  Amp_max << std::endl;
     if (optimize_population()) { 
        std::cout << "# optimizing population " << std::endl;
     } else { 
        std::cout << "# using temperature Tmax " <<  Tmax << std::endl;
     }
     std::cout << "# Gamma_max " <<  Gamma_max << std::endl;
     std::cout << "# NSPINS " << NSPINS << std::endl;
   }
};


int main()
{
    typedef Spin_From_Gene<SingleTriplet> Quintet_From_Gene;

    Quintet_From_Gene esr;
    esr.data.source_filename = "megaFreqPL12mVB161222.up";    
    esr.data.load_parameters();
    esr.data.Ymin = -100;
    esr.data.Ymax = 40;
    //    esr.data.Xmin = 800;
    //    esr.data.Xmax = 1500;
    esr.data.load_matrix(ODMR_Matrix::SIGNAL_X);
    esr.data.print_info();

    esr.Dmax = 3000;
    esr.Emax = esr.Dmax/3.0;
    esr.Amp_max = 300.0;
    esr.Gamma_max = 50.0;
    esr.Tmax = 2000.0;
    esr.print_info();

    simple_GA<Quintet_From_Gene> ga(esr);
    ga.initialize();
    ga.temp = 0.005;
    double anneal_eps = 0.7e-4;
    double temp_min = 0.0001;
    std::cout << "# anneal_eps "  << anneal_eps << std::endl;
    std::cout << "# temp_min "  << temp_min << std::endl;

    ga.print_info();
    ga.evaluate ();
    ga.report(-1);
    ga.keep_the_best();

    int MAXGENS = 100000;
    

    for (int generation = 0; generation < MAXGENS; generation++ ) {
       ga.temp *= (1.0 - anneal_eps);
       if (ga.temp < temp_min) ga.temp = temp_min;
       ga.step();
       ga.report(generation);
       if (!(generation % 20)) {
	  ga.print_best(generation);	  
       }
    }
    ga.print_best(MAXGENS);
}

