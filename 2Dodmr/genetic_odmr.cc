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

class ODMR_Matrix { 
    double Bmin, Bmax, FreqMin, FreqMax, dB, dFreq;
    int NX, NY;
    double dx, dy;
public : 
    std::string source_filename;      
    Eigen::MatrixXf odmr;    

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


    void load_matrix(bool use_sign = false) { 
       NY = ceil( (Ymax - Ymin) / dy);
       NX = ceil( (Xmax - Xmin) / dx);
       //       std::cerr << NX << std::endl;
       //       std::cerr << NY << std::endl;
       odmr = Eigen::MatrixXf::Zero(NX, NY);

       std::ifstream infile(source_filename.c_str());    
       std::string line;

       std::cout << "# load_matrix use sign " << use_sign << std::endl;
       while (std::getline(infile, line)) {  
	 std::istringstream iss(line);
	 double Freq,FreqBis,B,Icoil,vx,vy;
	 if ((iss >> Freq >> FreqBis >> B >> Icoil >> vx >> vy)) { 
	    double x = Freq * freq_scale;
	    double y = B * B_scale;
	    if (x >= Xmin && x < Xmax && y >= Ymin && y < Ymax) { 
	       if (use_sign) { 
		  odmr( enc_x(x), enc_y(y) ) = amp_scale * vx;	      
	       } else {
		  odmr( enc_x(x), enc_y(y) ) = amp_scale * sqrt(vx*vx+vy*vy);
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



class Triplet_From_Gene {
public: 
   enum TPenum { D, E, PHI, UZ, THETA, TEMP, AMP, GAMMA, NPARAMETERS } ;
   enum { NTRIPLETS = 4 };
   enum { NVARS = NTRIPLETS * NPARAMETERS };

   std::vector<SingleTriplet> triplets;
   std::vector<double> Amps;
   std::vector<double> signal;
   std::vector<double> rho0;

   ODMR_Matrix data;
   

   double Dmax;
   double Emax;
   double Tmax;
   double Amp_max;
   double Gamma_max;

   Triplet_From_Gene(void) : triplets(NTRIPLETS), Amps(NTRIPLETS) { 
   }

   void triplet_from_vector(int index, const std::vector<double> &pairvec, double Bz)
   {
       triplets[index].S.D = pairvec[D];
       triplets[index].S.E = pairvec[E];
       triplets[index].S.rot.angle_phi_uz(pairvec[THETA], pairvec[PHI], pairvec[UZ]);
       triplets[index].S.B << 0, 0, Bz;
       triplets[index].update_hamiltonian();
       triplets[index].diag(); 
   }


   double upper(int i) { 
      TPenum index = static_cast<TPenum>(i % NPARAMETERS);
      switch (index) {
         case D: 
	    return Dmax;
         case E:
	    return Emax;
         case UZ: 
	    return 1.0;
         case PHI: case THETA: 
	    return M_PI;
         case TEMP:
  	    return Tmax;
         case AMP:
  	    return Amp_max;
         case GAMMA:
	    return Gamma_max;
         default:
	    throw "case error: should not occur";
      }
   }

   double lower(int i) { 
      if (i == GAMMA) { return 0.0; } 
      else { return -upper(i); }
   }

   void update_triplets_from_gene_at_Bz(const double gene[], double Bz) { 
      std::vector<double> pairvec(NPARAMETERS);
      for (int i = 0; i < NTRIPLETS; i++) { 
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


   void comp_signal_for_omega(const double gene[]) {
      int NX = data.Xsize();
      signal.resize(data.Xsize());
      for (int i = 0; i < NX; i++) signal[i] = 0.0;

      for (int n = 0; n < NTRIPLETS; n++) { 
	 ESR_Signal<SingleTriplet> esr_from_triplets(triplets[n]);    
	 esr_from_triplets.update_from_spin_hamiltonian();
	 esr_from_triplets.gamma = gamma_from_gene(n, gene);
	 esr_from_triplets.gamma_diag = gamma_from_gene(n, gene);

	 if (rho0.size() == TripletPair::matrix_size) { 
	    esr_from_triplets.load_rho0(rho0);
	 } else { 
	   //	    odmr_from_triplets.load_rho0_thermal(1000.0);
	    esr_from_triplets.load_rho0_thermal(10000.0);
	 } 
	 for (int i = 0; i < NX; i++) { 
	   double omega = data.dec_x(i);
	   signal[i] += Amps[n] * imag(esr_from_triplets.chi1(omega));
	 }
      }
   }

   virtual double score(const double gene[]) { 
      int NB = data.Ysize();
      int Nomega = data.Xsize();

      double gene_score = 0.0;
      for (int b = 0; b < NB; b++) { 
	 double Bz = data.dec_y(b);
	 update_triplets_from_gene_at_Bz(gene, Bz);
	 comp_signal_for_omega(gene);
	 for (int w = 0; w < Nomega; w++) { 
	    gene_score += fabs( data.odmr(w, b) - signal[w] );
	 }
      }
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
      read_gene(filename, gene, NPARAMETERS * NTRIPLETS); 
   }

   void print_triplet_parameters(void) { 
      for (int i = 0; i < NTRIPLETS; i++) { 
	 std::cout << "# D " << triplets[i].S.D << std::endl;
	 std::cout << "# E " << triplets[i].S.E << std::endl;
	 std::cout << "# B " << triplets[i].S.B << std::endl;
	 std::cout << "# M " << triplets[i].S.rot.matrix()  << std::endl;
      }
  }


   void print_gene(const double gene[]) { 
      int NB = data.Ysize();
      int Nomega = data.Xsize();

      for (int b = 0; b < NB; b++) { 
	 double Bz = data.dec_y(b);	
	 update_triplets_from_gene_at_Bz(gene, Bz);
	 if (!b) { 
	    print_triplet_parameters();      
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
     std::cout << "# Dmax " <<  Dmax << std::endl;
     std::cout << "# Emax " <<  Emax << std::endl;
     std::cout << "# Amp_max " <<  Amp_max << std::endl;
     std::cout << "# Tmax " <<  Tmax << std::endl;
     std::cout << "# Gamma_max " <<  Gamma_max << std::endl;
     std::cout << "# NTRIPLETS " << NTRIPLETS << std::endl;
   }
};




class Triplet_Pair_From_Gene {
public: 
   enum TPenum { D1, D2, E1, E2, 
#ifdef OPTIMIZE_EXCHANGE
		 NJ, 
#endif
#ifdef OPTIMIZE_DIPOLE_DIPOLE
		 JDIP, PHI12, UZ12, 
#endif
		 PHI1, UZ1, THETA1, PHI2, UZ2, THETA2, AMP, GAMMA, 
		 NPARAMETERS } ;
   enum { NPAIRS = 1 };
   enum { NVARS = NPAIRS * NPARAMETERS };

   std::vector<TripletPair> triplets;
   std::vector<double> Amps;
   std::vector<double> signal;
   std::vector<double> rho0;

   ODMR_Matrix data;

   double Dmax;
   double Emax;
   double Jmax;
   double Jdip_max;
   double Amp_max;
   double Gamma_max;
   bool use_chi1;
   bool rescale;

   Triplet_Pair_From_Gene(void) : triplets(NPAIRS), Amps(NPAIRS) { 
      use_chi1 = true;
      rescale = true;
   }

   void triplet_pair_from_vector(int index, const std::vector<double> &pairvec, double Bz)
   {
      triplets[index].S1.D = pairvec[D1];
      triplets[index].S2.D = pairvec[D2];
      triplets[index].S1.E = pairvec[E1];
      triplets[index].S2.E = pairvec[E2];
#ifdef OPTIMIZE_EXCHANGE
      triplets[index].J = pairvec[NJ];
#else
      triplets[index].J = Jmax;
#endif
#ifdef OPTIMIZE_DIPOLE_DIPOLE
      triplets[index].Jdip = pairvec[JDIP];      
      double phi = pairvec[PHI12];
      double uz = pairvec[UZ12];    
      triplets[index].r12 << cos(phi) * sqrt(1. - uz*uz), sin(phi) * sqrt(1. - uz*uz), uz;
#endif       
      triplets[index].S1.rot.angle_phi_uz(pairvec[THETA1], pairvec[PHI1], pairvec[UZ1]);
      triplets[index].S2.rot.angle_phi_uz(pairvec[THETA2], pairvec[PHI2], pairvec[UZ2]);
      
      triplets[index].S1.B << 0, 0, Bz;
      triplets[index].S2.B << 0, 0, Bz;      

      triplets[index].update_hamiltonian();
      triplets[index].diag(); 
   }


   double upper(int i) { 
      TPenum index = static_cast<TPenum>(i % NPARAMETERS);
      switch (index) {
         case D1: 
         case D2:
	    return Dmax;
         case E1:
         case E2: 
	    return Emax;
#ifdef OPTIMIZE_EXCHANGE
         case NJ:
	    return Jmax;
#endif
#ifdef OPTIMIZE_DIPOLE_DIPOLE
         case JDIP: 	
	    return Jdip_max;
         case UZ12: 
	    return 1.0;
         case PHI12:
	    return M_PI;
#endif
         case UZ1: case UZ2: 
	    return 1.0;
         case PHI1: case PHI2: case THETA1: case THETA2:
	    return M_PI;
         case AMP:
  	    return Amp_max;
         case GAMMA:
	    return Gamma_max;
         default:
	    throw "case error: should not occur";
      }
   }

   double lower(int i) { 
      if (i == GAMMA) { return 0.0; } 
      else { return -upper(i); }
   }

   void update_triplets_from_gene_at_Bz(const double gene[], double Bz) { 
      std::vector<double> pairvec(NPARAMETERS);
      for (int i = 0; i < NPAIRS; i++) { 
	 for (int j = 0; j < NPARAMETERS; j++) { 
	    pairvec[j] = gene[i * NPARAMETERS + j];
	 }
	 triplet_pair_from_vector(i, pairvec, Bz);
	 Amps[i] = pairvec[AMP];
      }
   }

   double gamma_from_gene(int pair_number, const double gene[]) { 
       return gene[pair_number * NPARAMETERS + GAMMA];
   }


   void comp_signal_for_omega(const double gene[]) {
      int NX = data.Xsize();
      signal.resize(data.Xsize());
      for (int i = 0; i < NX; i++) signal[i] = 0.0;

      for (int n = 0; n < NPAIRS; n++) { 
	 ODMR_Signal<TripletPair> odmr_from_triplets(triplets[n]);    
	 odmr_from_triplets.update_from_spin_hamiltonian();
	 odmr_from_triplets.gamma = gamma_from_gene(n, gene);
	 odmr_from_triplets.gamma_diag = gamma_from_gene(n, gene);

	 if (rho0.size() == TripletPair::matrix_size) { 
	    odmr_from_triplets.load_rho0(rho0);
	 } else if (use_chi1) { 
	   //	    odmr_from_triplets.load_rho0_thermal(1000.0);
	    odmr_from_triplets.load_rho0_thermal(10000.0);
	 } else { 
	    odmr_from_triplets.load_rho0_from_singlet();
	 }

	 for (int i = 0; i < NX; i++) { 
	   double omega = data.dec_x(i);
	   signal[i] += (use_chi1) ? Amps[n] * imag(odmr_from_triplets.chi1(omega)) : Amps[n] * odmr_from_triplets.odmr(omega);
	 }
      }
   }

   virtual double score(const double gene[]) { 
      int NB = data.Ysize();
      int Nomega = data.Xsize();

      double gene_score = 0.0;
      for (int b = 0; b < NB; b++) { 
	 double Bz = data.dec_y(b);
	 update_triplets_from_gene_at_Bz(gene, Bz);
	 comp_signal_for_omega(gene);
	 for (int w = 0; w < Nomega; w++) { 
	    gene_score += fabs( data.odmr(w, b) - signal[w] );
	 }
      }
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
      read_gene(filename, gene, NPARAMETERS * NPAIRS); 
   }

   void print_triplet_parameters(void) { 
      std::cout << "# Ds " << triplets[0].S1.D << "   " << triplets[0].S2.D << std::endl;
      std::cout << "# Es " << triplets[0].S1.E << "   " << triplets[0].S2.E << std::endl;
      std::cout << "# B " << triplets[0].S1.B << "   " << triplets[0].S2.B << std::endl;
      std::cout << "# M1 " << triplets[0].S1.rot.matrix()  << std::endl;
      std::cout << "# M2 " << triplets[0].S2.rot.matrix()  << std::endl;
      std::cout << "# J " << triplets[0].J << std::endl;
      std::cout << "# Jdip " << triplets[0].Jdip << std::endl;
      std::cout << "# r12 " << triplets[0].r12 << std::endl;            
  }


   void print_gene(const double gene[]) { 
      int NB = data.Ysize();
      int Nomega = data.Xsize();

      for (int b = 0; b < NB; b++) { 
	 double Bz = data.dec_y(b);	
	 update_triplets_from_gene_at_Bz(gene, Bz);
	 if (!b) { 
	    print_triplet_parameters();      
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
     std::cout << "# Dmax " <<  Dmax << std::endl;
     std::cout << "# Emax " <<  Emax << std::endl;
#ifdef OPTIMIZE_EXCHANGE
     std::cout << "# Jmax " <<  Jmax << std::endl;
#else
     std::cout << "# J (exchange) fixed to " <<  Jmax << std::endl;
#endif
     std::cout << "# use_chi1 instead of odmr " << use_chi1 << std::endl;
#ifdef OPTIMIZE_DIPOLE_DIPOLE
     std::cout << "# Jdip_max " <<  Jdip_max << std::endl;
#else
     std::cout << "# Jdip_max disabled (= 0) " <<  std::endl;
#endif
     std::cout << "# Amp_max " <<  Amp_max << std::endl;
     std::cout << "# Gamma_max " <<  Gamma_max << std::endl;
     std::cout << "# NPAIRS " << NPAIRS << std::endl;
   }
};


struct Triplet_Pair_From_Population : Triplet_Pair_From_Gene { 
public : 
   enum { NVARS = TripletPair::matrix_size + 1 };
   enum { AMP = TripletPair::matrix_size };
   double gene0[Triplet_Pair_From_Gene::NVARS];

   Triplet_Pair_From_Population(void)  { 
      rho0.resize( TripletPair::matrix_size );
      if ( Triplet_Pair_From_Gene::NPAIRS != 1) { 
	 std::cerr << "# Triplet_Pair_From_Population designed for Triplet_Pair_From_Gene::NPAIRS = 1" << std::endl;
      }

   }

   void read_triplets(const char *filename) {  
      Triplet_Pair_From_Gene::read_gene(filename, gene0);
   }

   void read_population(const char *filename, double gene[]) {  
      Triplet_Pair_From_Gene::read_gene(filename, gene, NVARS);
   }



   void print_info(void) { 
      Triplet_Pair_From_Gene::print_info();
      std::cout << "# gene0 = [ " << std::endl;
      for (int i = 0; i < Triplet_Pair_From_Gene::NVARS; i++) { 
	std::cout << "#   " << gene0[i] << std::endl;	
      }
      std::cout << "# ] " << std::endl;
   }


   double upper(int i) { 
      if (i < AMP) return 1.0;
      else if (i == AMP) return Triplet_Pair_From_Gene::upper( Triplet_Pair_From_Gene::AMP );
   }

   double lower(int i) { 
      if (i < AMP) return 0.0;
      else if (i == AMP) return Triplet_Pair_From_Gene::lower( Triplet_Pair_From_Gene::AMP );
   }

   double score(const double gene[]) { 
      for (int i = 0; i < AMP; i++) { 
	 rho0[i] = gene[i];
      }      
      gene0[Triplet_Pair_From_Gene::AMP] = gene[AMP];
      return Triplet_Pair_From_Gene::score(gene0);

   }   

   void print_gene(const double gene[]) { 
      for (int i = 0; i < AMP; i++) { 
	 rho0[i] = gene[i];
      }      
      gene0[Triplet_Pair_From_Gene::AMP] = gene[AMP];
      Triplet_Pair_From_Gene::print_gene(gene0);
   }
};




template <class fitness_finder> struct genotype
{
  double gene[fitness_finder::NVARS];
  double score;
  double fitness;
  double upper[fitness_finder::NVARS];
  double lower[fitness_finder::NVARS];
  double rfitness;
  double cfitness;
};


//    Modified simple GA 
//    Original version by Dennis Cormier/Sita Raghavan/John Burkardt.
//    modified for C++
//  Reference:
//    Zbigniew Michalewicz,
//    Genetic Algorithms + Data Structures = Evolution Programs,
//    Third Edition,
//    Springer, 1996,
//    ISBN: 3-540-60676-9,
//    LC: QA76.618.M53.
//

template <class fitness_finder> class simple_GA { 
   fitness_finder &ffinder;
   std::vector<fitness_finder> ffinder_list;
    
   int int_uniform_ab ( int a, int b ) { 
      return a + (rand() % (b - a + 1));
   }

   double real_uniform_ab ( double a, double b ) { 
      return a + (b - a) * (double) rand() / (double) RAND_MAX;
   }

   void Xover ( int one, int two ) {
      //  Select the crossover point.
      int point = int_uniform_ab ( 0, fitness_finder::NVARS - 1 );
      //  Swap genes in positions 0 through POINT-1.
      for (int i = 0; i < point; i++ ) {
	 double t = population[one].gene[i];
	 population[one].gene[i] = population[two].gene[i];
	 population[two].gene[i] = t;
      }
   }

   void copy_gene(int from, int to) { 
      for (int i = 0; i < fitness_finder::NVARS; i++ ) {
        population[to].gene[i] = population[from].gene[i];
      }
      population[to].score = population[from].score;
      population[to].fitness = population[from].fitness;      
   }

public : 
   enum { POPSIZE = 32 };


   double PXOVER = 0.8;
   double PMUTATION = 0.1;
   struct genotype<fitness_finder> population[POPSIZE+1];
   struct genotype<fitness_finder> newpopulation[POPSIZE+1]; 
   double temp;

   simple_GA(fitness_finder &f) : ffinder(f) {
      PXOVER = 0.8;
      PMUTATION = 0.1;
      ffinder_list.reserve(POPSIZE);
      for (int i = 0; i < POPSIZE; i++) { 
	ffinder_list.push_back( f );
      }
   }

   void crossover (void) {
      const double a = 0.0;
      const double b = 1.0;
      int mem;
      int one;
      int first = 0;
      
      for ( mem = 0; mem < POPSIZE; ++mem ) {
	double x = real_uniform_ab ( a, b );
	
	if ( x < PXOVER ) {
	  ++first;
	  
	  if ( first % 2 == 0 ) {
	    Xover ( one, mem );
	  } else {
	    one = mem;
	  }
	}
      }
      return;
   }

// 
//  If the best individual from the new population is better than 
//  the best individual from the previous population, then 
//  copy the best from the new population; else replace the 
//  worst individual from the current population with the 
//  best one from the previous generation                     
//  
   void elitist(void) {
     int i;
     double best, worst;
     int best_mem, worst_mem;
//
//
// elitist based on scores and not fitness since scores are temperature independent. 
//   
     best = worst = population[0].score;
     best_mem = worst_mem = 0;

     for (i = 0; i < POPSIZE - 1; ++i) {
        if ( population[i+1].score < population[i].score ) {
	   if ( best <= population[i].score ) {
	      best = population[i].score;
	      best_mem = i;
	   }
	   
	   if ( population[i+1].score <= worst ) {
	      worst = population[i+1].score;
	      worst_mem = i + 1;
	   }
	} else {
	  if ( population[i].score <= worst ) {
	     worst = population[i].score;
	     worst_mem = i;
	  }
	  if ( best <= population[i+1].score ) {
	     best = population[i+1].score;
	     best_mem = i + 1;
	  }
	}
     }

     if ( population[POPSIZE].score <= best ) {
        copy_gene(best_mem, POPSIZE); 
     } else {
        copy_gene(POPSIZE, worst_mem);
     } 
   }


   void evaluate(void) {
      // when we are dealing with random fitnesses we recompute the fitness of the best individual at POPSIZE 
      // not now 
      #pragma omp parallel for // num_threads(4)
      for (int member = 0; member < POPSIZE; member++ ) { 
	 population[member].score = ffinder_list[member].score(population[member].gene);	 
      }
      double avscore = 0.0;
      for (int member = 0; member < POPSIZE; member++ ) { 
	 avscore += population[member].score/(double) (POPSIZE+1);
      }
      for (int member = 0; member < POPSIZE; member++ ) { 
	// towards minimum
	// 	 population[member].fitness = exp ( -(population[member].score - avscore)/temp );
	// towards maximum
	population[member].fitness = exp ( (population[member].score - avscore)/temp );
	//	 population[member].fitness = population[member].score;
      }
   }

   void initialize (void) {
      for (int j = 0; j <= POPSIZE; j++ ) {
	 population[j].fitness = 0;	
	 population[j].score = 0;	
	 population[j].rfitness = 0;
	 population[j].cfitness = 0;
         for (int i = 0; i < fitness_finder::NVARS; i++ ) {
	    population[j].lower[i] = ffinder.lower(i);
	    population[j].upper[i] = ffinder.upper(i);
	    population[j].gene[i] = real_uniform_ab (population[j].lower[i], population[j].upper[i]); 
	 }
      }
   }  

   void initial_values(int num, const double *gene) { 
      std::cout << "# setting initial_value for gene " << num << std::endl;
      for (int i = 0; i < fitness_finder::NVARS; i++ ) {
         population[num].gene[i] = gene[i];
      }
      std::cout << "# with initial score " << ffinder.score(population[num].gene) << std::endl;
   }

   void keep_the_best (void) { 
      int cur_best;
      int mem;

      cur_best = 0;
      population[POPSIZE].fitness = 0;
      
      for ( mem = 0; mem < POPSIZE; mem++ ) {
        if ( population[POPSIZE].fitness < population[mem].fitness ) {
	  cur_best = mem;
	  population[POPSIZE].fitness = population[mem].fitness;
	}
      }
      // 
      //  Once the best member in the population is found, copy the genes.
      //
      copy_gene(cur_best, POPSIZE);
      return;
   }


   void mutate (void) { 
      const double a = 0.0;
      const double b = 1.0;
      double lbound;
      double ubound;
      double x;

      for (int i = 0; i < POPSIZE; i++ ) {
	 for (int j = 0; j < fitness_finder::NVARS; j++ ) {
	    x = real_uniform_ab (a, b);
	    if ( x < PMUTATION ) {	      
	       lbound = population[i].lower[j];
	       ubound = population[i].upper[j];
	       population[i].gene[j] = real_uniform_ab (lbound, ubound);
	    }
	 }
      }
   }
  

   void mutate (double amplitude) { 
      const double a = 0.0;
      const double b = 1.0;
      double lbound;
      double ubound;
      double x;

      for (int i = 0; i < POPSIZE; i++ ) {
	 for (int j = 0; j < fitness_finder::NVARS; j++ ) {
	    x = real_uniform_ab (a, b);
	    if ( x < PMUTATION ) {	      
	       lbound = std::max(population[i].lower[j], population[i].gene[j] - amplitude);
	       ubound = std::min(population[i].upper[j], population[i].gene[j] + amplitude);
	       population[i].gene[j] = real_uniform_ab (lbound, ubound);
	    }
	 }
      }
   }

   void report ( int generation ) {
      double avg;
      double best_val;
      double square_sum;
      double stddev;
      double sum;
      double sum_square;
      double av_score; 

      if ( generation == 0 ) {
	 std::cout << "\n";
	 std::cout << "Value     Generation    Best         Best       Average    Average    Standard \n";
	 std::cout << "Value     number        value        Score      fitness    score      deviation \n";
	 std::cout << "\n";
      }

      sum = 0.0;
      sum_square = 0.0;
      av_score = 0.0;

      for (int i = 0; i < POPSIZE; i++ ) {
	 sum += population[i].fitness;
	 sum_square += population[i].fitness * population[i].fitness;
	 av_score += population[i].score;
      }

      avg = sum / ( double ) POPSIZE;
      av_score /= (double) POPSIZE;
      square_sum = avg * avg * POPSIZE;
      stddev = sqrt ( ( sum_square - square_sum ) / ( POPSIZE - 1 ) );
      best_val = population[POPSIZE].fitness;
      double best_score = population[POPSIZE].score;

      std::cout << "  " << std::setw(8) << "equal " 
                << "  " << std::setw(8) << generation 
		<< "  " << std::setw(14) << best_val 
		<< "  " << std::setw(14) << best_score
		<< "  " << std::setw(14) << avg 
		<< "  " << std::setw(14) << av_score	
		<< "  " << std::setw(14) << stddev << "\n";

      std::cout << std::flush;
   }

 
   void selector (void) {
      const double a = 0.0;
      const double b = 1.0;
      int i;
      int j;
      int mem;
      double p;
      double sum;
      //
      //  Find the total fitness of the population.
      //
      sum = 0.0;
      for ( mem = 0; mem < POPSIZE; mem++ ) {
	 sum = sum + population[mem].fitness;
      }
      //
      //  Calculate the relative fitness of each member.
      //
      for ( mem = 0; mem < POPSIZE; mem++ ) {
	 population[mem].rfitness = population[mem].fitness / sum;
      }
      // 
      //  Calculate the cumulative fitness.
      //
      population[0].cfitness = population[0].rfitness;
      for ( mem = 1; mem < POPSIZE; mem++ ) {
	 population[mem].cfitness = population[mem-1].cfitness + population[mem].rfitness;
      }
      // 
      //  Select survivors using cumulative fitness. 
      //
      for ( i = 0; i < POPSIZE; i++ ) { 
	 p = real_uniform_ab (a, b);
	 if ( p < population[0].cfitness ) {
	    newpopulation[i] = population[0];      
	 } else {
	    // could use a dichotomic search - 
	    for ( j = 0; j < POPSIZE; j++ ) { 
	       if ( population[j].cfitness <= p && p < population[j+1].cfitness ) {
		  newpopulation[i] = population[j+1]; 
	       }
	    }
	 }
      }
      // 
      //  Overwrite the old population with the new one.
      //
      for ( i = 0; i < POPSIZE; i++ ) {
	 population[i] = newpopulation[i]; 
      }

      return;     
   }

   void step(void) { 
       selector();
       crossover();
       mutate();
       evaluate();
       elitist();
   }

   void print_info(void) { 
      std::cout << "# POPSIXE " << POPSIZE << std::endl;
      std::cout << "# NVARS " << fitness_finder::NVARS << std::endl;
      std::cout << "# PXOVER " << PXOVER << std::endl;
      std::cout << "# PMUTATION " << PMUTATION << std::endl;
      std::cout << "# temp " << temp << std::endl;
   }

   void print_best(int generation) { 
      std::cout << "# best gene = " << population[POPSIZE].fitness << "\n";
      for (int i = 0; i < fitness_finder::NVARS; i++ ) {
         std::cout << generation << "   " << i << "    " << population[POPSIZE].gene[i] << "  %" << std::endl;
      }
      std::cout << "# with fitness = " << population[POPSIZE].fitness << "\n";
   }
};

int main()
{
    Triplet_From_Gene esr;
    esr.data.source_filename = "megaFreqPL12mVB161222.up";    
    esr.data.load_parameters();
    esr.data.Ymin = -100;
    esr.data.Ymax = 40;
    //    esr.data.Xmin = 800;
    //    esr.data.Xmax = 1500;
    esr.data.load_matrix();
    esr.data.print_info();

    esr.Dmax = 2000;
    esr.Emax = esr.Dmax/3.0;
    esr.Amp_max = 100.0;
    esr.Gamma_max = 50.0;
    esr.Tmax = 2000.0;
    esr.print_info();

    simple_GA<Triplet_From_Gene> ga(esr);
    ga.initialize();
    ga.temp = 0.005;
    double anneal_eps = 0.7e-4;
    double temp_min = 0.0001;
    std::cout << "# anneal_eps "  << anneal_eps << std::endl;
    std::cout << "# temp_min "  << temp_min << std::endl;

    double gene0[Triplet_From_Gene::NVARS];
    esr.read_gene("opt4T.gene", gene0);
    std::cout << "# gene0 score " << esr.score(gene0) << std::endl;
    esr.print_gene(gene0);
    exit(0);
    ga.initial_values(0, gene0);
        
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

int main_ch1eq()
{
    Triplet_Pair_From_Gene odmr;
    odmr.data.source_filename = "megaFreqPL12mVB161222.up";    
    odmr.data.load_parameters();
    odmr.data.Ymin = -100;
    odmr.data.Ymax = 40;
    //    odmr.data.Xmin = 800;
    //    odmr.data.Xmax = 1500;
    odmr.data.load_matrix();
    odmr.data.print_info();

    odmr.Dmax = 2000;
    odmr.Emax = odmr.Dmax/3.0;
    odmr.Jmax = 10000;
    odmr.Jdip_max = odmr.Dmax;
    odmr.Amp_max = 100.0;
    odmr.Gamma_max = 50.0;
    odmr.use_chi1 = true;
    odmr.print_info();

    /**
    double gene0[Triplet_Pair_From_Gene::NVARS];
    odmr.read_gene("optNoDip.gene", gene0);
    std::cout << "# gene0 score " << odmr.score(gene0) << std::endl;
    odmr.print_gene(gene0);
    exit(0);
    **/

    simple_GA<Triplet_Pair_From_Gene> ga(odmr);
    ga.initialize();
    //    ga.initial_values(17, gene0);
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




int main_pop()
{
    Triplet_Pair_From_Population odmr;
    odmr.data.source_filename = "megaFreqPL12mVB161222.up";    
    odmr.data.load_parameters();
    odmr.data.Ymin = -100;
    odmr.data.Ymax = 40;
    //    odmr.data.Xmin = 800;
    //    odmr.data.Xmax = 1500;
    odmr.data.load_matrix(true);
    odmr.data.print_info();

    odmr.Dmax = 2000;
    odmr.Emax = odmr.Dmax/3.0;
    odmr.Jmax = 2000;
    odmr.Jdip_max = odmr.Dmax;
    odmr.Amp_max = 300.0;
    odmr.Gamma_max = 50.0;
    odmr.use_chi1 = true;

    odmr.read_triplets("opt2.gene");

    double gene0[Triplet_Pair_From_Population::NVARS];
    odmr.read_population("opt2Pop.gene", gene0);
    for (int i = 0; i < Triplet_Pair_From_Population::NVARS; i++) { 
      std::cerr << gene0[i] << std::endl;
    }
    std::cout << "# gene0 score " << odmr.score(gene0) << std::endl;
    odmr.print_gene(gene0);
    exit(0);



    odmr.print_info();


    simple_GA<Triplet_Pair_From_Population> ga(odmr);
    ga.initialize();
    //    ga.initial_values(17, gene0);
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
