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
#include <Eigen/Sparse>

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


    double B_scale;
    double freq_scale;
    double amp_scale;
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
       NY = floor( (Ymax - Ymin) / dy) + 1;
       NX = floor( (Xmax - Xmin) / dx) + 1;
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

struct Spin_From_Gene_Base {
   enum TPenum { D, E, PHI, UZ, THETA, AMP, GAMMA, NTP } ;
   virtual double score(const double gene[]) = 0;
   virtual void print_info(void) = 0;
};

template <int NSPINS, class Spin> class Spin_From_Gene : public Spin_From_Gene_Base {

   //MatrixXd signal;
public: 
   MatrixXd signal;
  // optimize population
   enum { POPSTART = NTP };
   enum { POPEND = NTP  + Spin::matrix_size };
   enum { NPARAMETERS = POPEND };
  
  // don't optimize population 
  //   enum { NPARAMETERS = NTP };

   bool inline optimize_population(void) { 
       return (NPARAMETERS - NTP == Spin::matrix_size);
   }

   enum { NVARS = NSPINS * NPARAMETERS };

   ODMR_Matrix data;
   
   double Dmax;
   double Emax;
   double Tmax;
   double Amp_max;
   double Gamma_max;

   int spin_index(int pair_number, int entry) { 
       return pair_number * NPARAMETERS + entry;
   }

private:
   std::vector<Spin,  Eigen::aligned_allocator<Spin> > spins;
   std::vector<double> Amps;
   std::vector<double> varmin;
   std::vector<double> varmax;
   std::vector<double> rho0;   
public:  

  
  Spin_From_Gene<NSPINS,Spin>(void) : spins(NSPINS), Amps(NSPINS), varmin(NVARS), varmax(NVARS), rho0(Spin::matrix_size) { 
      for (int n = 0; n < NSPINS; n++) { 
	 varmin[ spin_index(n, AMP) ] = 0.0;
	 varmin[ spin_index(n, GAMMA) ] = 0.0;

	 varmin[ spin_index(n, UZ) ] = -1.0;
	 varmin[ spin_index(n, THETA) ] = -M_PI;
	 varmin[ spin_index(n, PHI) ] = -M_PI;
	 for (int j = POPSTART; j < POPEND; j++) { 
	    varmin[ spin_index(n, j) ] = 0.0;
	 }

	 varmax[ spin_index(n, UZ) ] = 1.0;
	 varmax[ spin_index(n, THETA) ] = M_PI;
	 varmax[ spin_index(n, PHI) ] = M_PI;
	 for (int j = POPSTART; j < POPEND; j++) { 
	    varmax[ spin_index(n, j) ] = 1.0;
	 }
      }
      std::cout << "FINISHED NSPINS = " << NSPINS << std::endl;
   }

   void set_min_for_all_spins(int index, double value) { 
      for (int n = 0; n < NSPINS; n++) { 
	 varmin[ spin_index(n, index) ] = value;
      }  
   }

   void set_max_for_all_spins(int index, double value) { 
      for (int n = 0; n < NSPINS; n++) { 
	 varmax[ spin_index(n, index) ] = value;
      }  
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
	 std::cout << "# eigenvalues ";
	 for (int j = 0; j < Spin::matrix_size; j++) {
	   std::cout << spins[i].eval(j) << "   ";
	 }
	 std::cout << std::endl;
      }
  }

   double upper(int i) { 
      return varmax[i];
   }

   double lower(int i) { 
      return varmin[i];
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

   void pop_from_gene(int pair_number, const double gene[]) { 
       for (int i = NTP; i < NPARAMETERS; i++) { 
	  rho0[i - NTP] = gene[ spin_index(pair_number, i) ];
       }
   }

   void comp_signal_for_omega(const double gene[]) {
      int NX = data.Xsize();
      signal.resize(NSPINS+1, data.Xsize());
      signal.setZero();

      for (int n = 0; n < NSPINS; n++) { 
	 ESR_Signal<Spin> esr_from_spins(spins[n]);    
	 esr_from_spins.update_from_spin_hamiltonian();
	 esr_from_spins.gamma = gene[ spin_index(n, GAMMA) ]; 
	 esr_from_spins.gamma_diag = gene[ spin_index(n, GAMMA) ]; 

	 if (optimize_population()) { 
	    pop_from_gene(n, gene);
	    esr_from_spins.load_rho0(rho0);
	 } else { 
	    esr_from_spins.load_rho0_thermal(Tmax);
	 }

	 for (int i = 0; i < NX; i++) { 
	   double omega = data.dec_x(i);
	   signal(0, i) += Amps[n] * imag(esr_from_spins.chi1(omega));
	   signal(n+1, i) = Amps[n] * imag(esr_from_spins.chi1(omega));
	 }
      }
   }

   double score(const double gene[]) { 
      int NB = data.Ysize();
      int Nomega = data.Xsize();

      double gene_score = 0.0;
      for (int b = 0; b < NB; b++) { 
	 double Bz = data.dec_y(b);
	 update_spins_from_gene_at_Bz(gene, Bz);
	 comp_signal_for_omega(gene);
	 for (int w = 0; w < Nomega; w++) { 
	    gene_score += fabs( data.odmr(w, b) - signal(0, w) );
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
	    i++;
	 }
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
      
      std::cout << "# gene score " << score(gene) << std::endl;

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
	    std::cout << omega << "   " << Bz << "   " <<  data.odmr(w, b) << "   " << signal(0, w) << "   ";
	    for (int n = 0; n < NSPINS; n++) { 
	        std::cout << signal(n+1, w) << "   ";
	    }
	    std::cout << std::endl;
	 }
	 std::cout << std::endl;
      }
   }

   void print_info(void) { 
     std::cout << "# spin_matrix_size " << Spin::matrix_size << std::endl;
     std::cout << "# NSPINS " << NSPINS << std::endl;
     std::cout << "# variables min/max values: " << std::endl;
     for (int i = 0; i < NVARS; i++) { 
       std::cout << "# " << varmin[i] << "   " << varmax[i] << std::endl;
     }
   }
  
   void print_gene_parameters(const double gene[]) { 
      std::cout << "# printing gene with population " << std::endl;
      for (int n = 0; n < NSPINS; n++) { 
	 for (int index = 0; index < NPARAMETERS; index++) { 
	    std::cout << "#" << 0 << "   " << index << "   " << gene[n * NPARAMETERS + index] << std::endl;
	 }
      }
   }

};


template <int Ntriplets, int Nquintets> class Triplet_Quintet_From_Gene : public Spin_From_Gene_Base { 
public:
    typedef Spin_From_Gene<Ntriplets, SingleTriplet> Triplets;
    //  typedef Spin_From_Gene<Nquintets, SingleQuintet> Quintets;
    typedef Spin_From_Gene<Nquintets, SingleQuintet> Quintets;  
    enum { NvarsT = Triplets::NVARS };
    enum { NvarsQ = Quintets::NVARS };
    enum { NS3 = Ntriplets, NS5 = Nquintets };
    enum { NVARS = NvarsT + NvarsQ} ;
private:
    const double *get_triplet_gene(const double *gene) {
       return gene;
    }

    const double *get_quintet_gene(const double *gene) {
       return &gene[NvarsT];
    }
public :


    Triplets triplets;
    Quintets quintets;
    ODMR_Matrix &data;

    Triplet_Quintet_From_Gene<Ntriplets, Nquintets>(void) : triplets(), quintets(), data(triplets.data) { 
    }
  
    void load_matrix() {
       triplets.data.load_matrix(ODMR_Matrix::SIGNAL_X);
       quintets.data = triplets.data;
    }

    void set_max_for_triplets(int num, double value) { 
       triplets.set_max_for_all_spins(num,  value );//439.5 );
    }
  
  void set_min_for_triplets(int num, double value) { 
    triplets.set_min_for_all_spins(num,  value );//439.5 );
  }
  
  void set_max_for_quintets(int num, double value) {
    quintets.set_max_for_all_spins( num,  value );
  }

  void set_min_for_quintets(int num, double value) {
    quintets.set_min_for_all_spins( num,  value );
  }


  void read_gene(const char *filename, double gene[]) {
    triplets.Triplets::read_gene(filename, gene, NVARS);
  }

   double upper(int i) { 
      if (i < NvarsT) return triplets.Triplets::upper(i);
      else return quintets.upper(i-NvarsT);
   }

   double lower(int i) { 
      if (i < NvarsT) return triplets.Triplets::lower(i);
      else return quintets.lower(i-NvarsT);
   }

   int triplet_index(int pair_number, int entry) { 
      return pair_number * Triplets::NPARAMETERS + entry;
   }

   int quintet_index(int pair_number, int entry) { 
      return NvarsT + pair_number * Quintets::NPARAMETERS + entry;
   }
  
  
  double score(const double gene[]) {
    const double *gene_triplet = get_triplet_gene(gene);
    const double *gene_quintet = get_quintet_gene(gene);
    /****
    std::cerr << "# NvarsT " << NvarsT << std::endl;
    std::cerr << "# NvarsQ " << NvarsQ << std::endl;
    for (int i = 0; i < NvarsT; i++) { 
      std::cerr << i << "    " << gene_triplet[i] << " T " << gene[i] << std::endl;
    }

    for (int i = 0; i < NvarsQ; i++) { 
      std::cerr << i << "    " << gene_quintet[i] << " Q " << gene[i + NvarsT] << std::endl;
    }
    ****/
    int NB = triplets.data.Ysize();
    int Nomega = triplets.data.Xsize();
    double gene_score = 0.0;
    for (int b = 0; b < NB; b++) { 
      double Bz = triplets.data.dec_y(b);
      triplets.update_spins_from_gene_at_Bz(gene_triplet, Bz);
      quintets.update_spins_from_gene_at_Bz(gene_quintet, Bz) ;
      triplets.comp_signal_for_omega(gene_triplet);
      quintets.comp_signal_for_omega(gene_quintet);
      for (int w = 0; w < Nomega; w++) { 
          gene_score += fabs( triplets.data.odmr(w, b) - triplets.signal(0, w) - quintets.signal(0,w) );
      }
    }
    //    std::cerr << "# score quintet << " << quintets.score( gene_quintet ) << std::endl;
    //    std::cerr << "# score << " << 1.0/gene_score << std::endl;
    return 1.0 / gene_score;   
  }


   void print_gene(const double gene[]) { 
      int NB = data.Ysize();
      int Nomega = data.Xsize();

      const double *gene_triplet = get_triplet_gene(gene);
      const double *gene_quintet = get_quintet_gene(gene);
      
      std::cout << "# gene score " << score(gene) << std::endl;

      for (int b = 0; b < NB; b++) { 
	 double Bz = data.dec_y(b);	
	 triplets.update_spins_from_gene_at_Bz(gene_triplet, Bz);
	 quintets.update_spins_from_gene_at_Bz(gene_quintet, Bz);
	 if (!b) { 
	    triplets.print_gene_parameters(gene_triplet);      
	    quintets.print_gene_parameters(gene_quintet);      
	 }

	 triplets.comp_signal_for_omega(gene);
	 quintets.comp_signal_for_omega(gene);
	 for (int w = 0; w < Nomega; w++) { 
	    double omega = data.dec_x(w);
	    std::cout << omega << "   " << Bz << "   " <<  data.odmr(w, b) << "   " << triplets.signal(0, w) + quintets.signal(0, w)<< "   ";
	    for (int n = 0; n < Ntriplets; n++) { 
	        std::cout << triplets.signal(n+1, w) << "   ";
	    }
	    for (int n = 0; n < Nquintets; n++) { 
	        std::cout << quintets.signal(n+1, w) << "   ";
	    }
	    std::cout << std::endl;
	 }
	 std::cout << std::endl;
      }
   }


  void print_info(void) { 
    triplets.print_info();
    quintets.print_info();
  }
}; 



int main()
{
    typedef Triplet_Quintet_From_Gene < 4, 1 > Triplet_Quintet;
    Triplet_Quintet sumTQ;

    sumTQ.data.source_filename = "megaFreqPL12mVB161222.up";
    sumTQ.data.load_parameters();
    sumTQ.data.Ymin = -100;
    sumTQ.data.Ymax = 40;
    //    esr.data.Xmin = 1120;
    //    esr.data.Xmax = 1400;
    sumTQ.load_matrix();

    // D, E, PHI, UZ, THETA, AMP, GAMMA, NTP
    sumTQ.set_max_for_triplets(Triplet_Quintet::D, 2500.0);
    sumTQ.set_min_for_triplets(Triplet_Quintet::D, 0);
    sumTQ.set_max_for_triplets(Triplet_Quintet::E, 1500.0);
    sumTQ.set_min_for_triplets(Triplet_Quintet::E, 0);
    sumTQ.set_max_for_triplets(Triplet_Quintet::AMP, 400);
    sumTQ.set_max_for_triplets(Triplet_Quintet::GAMMA, 40);
    sumTQ.set_max_for_quintets(Triplet_Quintet::D, 439.5);
    sumTQ.set_min_for_quintets(Triplet_Quintet::D, 439.5);
    sumTQ.set_max_for_quintets(Triplet_Quintet::E, 30.8);
    sumTQ.set_min_for_quintets(Triplet_Quintet::E, 30.8);
    sumTQ.set_max_for_quintets(Triplet_Quintet::AMP, 600);
    sumTQ.set_max_for_quintets(Triplet_Quintet::GAMMA, 60);

    double gene0[ Triplet_Quintet::NVARS ]; 
    //    sumTQ.read_gene("TQtogether.gene", gene0);
    

    sumTQ.data.print_info();
    sumTQ.print_info();

    //    double gene0[ Triplet_Quintet::NVARS ]; 
    sumTQ.read_gene("4trip1quinvar1.gene", gene0);
    sumTQ.print_gene(gene0);
    exit(0);

    simple_GA <Triplet_Quintet> ga(sumTQ);
    ga.initialize();
    ga.initial_values(0, gene0);
    ga.fix_to_gene(gene0);

    for (int nt = 0; nt < sumTQ.NS3; nt++) { 
       ga.unfix( sumTQ.triplet_index(nt, Triplet_Quintet::Triplets::AMP) );
       ga.unfix( sumTQ.triplet_index(nt, Triplet_Quintet::Triplets::GAMMA) );
       for (int index = Triplet_Quintet::Triplets::POPSTART; index < Triplet_Quintet::Triplets::POPEND; index++) { 
	  ga.unfix( sumTQ.triplet_index(nt, index) );
       }
       ga.unfix( sumTQ.triplet_index(nt, Triplet_Quintet::Triplets::THETA) );
       ga.unfix( sumTQ.triplet_index(nt, Triplet_Quintet::Triplets::PHI) );
       ga.unfix( sumTQ.triplet_index(nt, Triplet_Quintet::Triplets::UZ) );
    }

    for (int nq = 0; nq < sumTQ.NS5; nq++) { 
       ga.unfix( sumTQ.quintet_index(nq, Triplet_Quintet::Quintets::AMP) );
       ga.unfix( sumTQ.quintet_index(nq, Triplet_Quintet::Quintets::GAMMA) );
       for (int index = Triplet_Quintet::Quintets::POPSTART; index < Triplet_Quintet::Quintets::POPEND; index++) { 
	  ga.unfix( sumTQ.quintet_index(nq, index) );
       }
       ga.unfix( sumTQ.quintet_index(nq, Triplet_Quintet::Quintets::THETA) );
       ga.unfix( sumTQ.quintet_index(nq, Triplet_Quintet::Quintets::PHI) );
       ga.unfix( sumTQ.quintet_index(nq, Triplet_Quintet::Quintets::UZ) );
    }
   
    ga.temp = 0.005;

    double anneal_eps = 0.7e-4;
    double temp_min = 0.0001;
    std::cout << "# anneal_eps "  << anneal_eps << std::endl;
    std::cout << "# temp_min "  << temp_min << std::endl;

    ga.print_info();
    ga.evaluate ();
    ga.report(-1);
    ga.keep_the_best();

    int MAXGENS = 2000000;
    

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

