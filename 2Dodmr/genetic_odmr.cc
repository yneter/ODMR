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

#include "../odmr_triplets.cc"

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


    void load_matrix(void) { 
       NY = ceil( (Ymax - Ymin) / dy);
       NX = ceil( (Xmax - Xmin) / dx);
       //       std::cerr << NX << std::endl;
       //       std::cerr << NY << std::endl;
       odmr = Eigen::MatrixXf::Zero(NX, NY);

       std::ifstream infile(source_filename.c_str());    
       std::string line;

       while (std::getline(infile, line)) {  
	 std::istringstream iss(line);
	 double Freq,FreqBis,B,Icoil,vx,vy;
	 if ((iss >> Freq >> FreqBis >> B >> Icoil >> vx >> vy)) { 
	    double x = Freq * freq_scale;
	    double y = B * B_scale;
	    if (x >= Xmin && x < Xmax && y >= Ymin && y < Ymax) { 
	       odmr( enc_x(x), enc_y(y) ) = amp_scale * sqrt(vx*vx+vy*vy);
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




class Triplet_Pair_From_Gene {
   std::vector<TripletPair> triplets;
   std::vector<double> Amps;
   std::vector<double> signal;
public: 
#ifdef OPTIMIZE_DIPOLE_DIPOLE
   enum TPenum { D1, D2, E1, E2, NJ, JDIP, PHI12, UZ12, PHI1, UZ1, THETA1, PHI2, UZ2, THETA2, AMP, GAMMA, N_TP } ;
#else
   enum TPenum { D1, D2, E1, E2, NJ, PHI1, UZ1, THETA1, PHI2, UZ2, THETA2, AMP, GAMMA, N_TP } ;
#endif 
   static const int NPAIRS;

   ODMR_Matrix data;
   double Dmax;
   double Emax;
   double Jmax;
   double Jdip_max;
   double Amp_max;
   double Gamma_max;
   bool use_chi1;


   Triplet_Pair_From_Gene(void) : triplets(NPAIRS), Amps(NPAIRS) { 
      use_chi1 = true;
   }

   void triplet_pair_from_vector(int index, const std::vector<double> &pairvec, double Bz)
   {
      triplets[index].S1.D = pairvec[D1];
      triplets[index].S2.D = pairvec[D2];
      triplets[index].S1.E = pairvec[E1];
      triplets[index].S2.E = pairvec[E2];
      triplets[index].J = pairvec[NJ];
      
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
      TPenum index = static_cast<TPenum>(i % N_TP);
      switch (index) {
         case D1: 
         case D2:
	    return Dmax;
         case E1:
         case E2: 
	    return Emax;
         case NJ:
	    return Jmax;
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
	    throw "case error: N_TP should not occur";
      }
   }

   double lower(int i) { 
      if (i == GAMMA) { return 0.0; } 
      else { return -upper(i); }
   }

private:
   void update_triplets_from_gene_at_Bz(const double gene[], double Bz) { 
      std::vector<double> pairvec(N_TP);
      for (int i = 0; i < NPAIRS; i++) { 
	 for (int j = 0; j < N_TP; j++) { 
	    pairvec[j] = gene[i * N_TP + j];
	 }
	 triplet_pair_from_vector(i, pairvec, Bz);
	 Amps[i] = pairvec[AMP];
      }
   }


   void comp_signal_for_omega(const double gene[]) {
      int NX = data.Xsize();
      signal.resize(data.Xsize());
      for (int i = 0; i < NX; i++) signal[i] = 0.0;

      for (int n = 0; n < NPAIRS; n++) { 
	 ODMR_Signal<TripletPair> odmr_from_triplets(triplets[n]);    
	 odmr_from_triplets.update_from_spin_hamiltonian();
	 odmr_from_triplets.gamma = gene[n * N_TP + GAMMA];
	 odmr_from_triplets.gamma_diag = gene[n * N_TP + GAMMA];

	 if (use_chi1) { 
	    odmr_from_triplets.load_rho0_thermal(1000.0);
	    for (int i = 0; i < NX; i++) { 
	       double omega = data.dec_x(i);
	       signal[i] += Amps[n] * imag(odmr_from_triplets.chi1(omega));
	    }
	 } else { 
	    odmr_from_triplets.load_rho0_from_singlet();
	    for (int i = 0; i < NX; i++) { 
	       double omega = data.dec_x(i);
	       //	    signal[i] += Amps[n] * imag(odmr_from_triplets.chi1(omega));
	       signal[i] += Amps[n] * odmr_from_triplets.odmr(omega);
	    }
	 }
      }
   }

public:
   double score(const double gene[]) { 
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

   void read_gene(const char *filename, double gene[]) {  
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
       if (i != N_TP * NPAIRS) { 
	  std::cerr << "read_gene - incorrect read " << i << "  " << N_TP * NPAIRS << std::endl;
       }
   }

   void print_gene(const double gene[]) { 
      int NB = data.Ysize();
      int Nomega = data.Xsize();

      for (int b = 0; b < NB; b++) { 
	 double Bz = data.dec_y(b);	
	 update_triplets_from_gene_at_Bz(gene, Bz);
	 std::cout << "# Bz " << Bz << std::endl;
	 std::cout << "# Ds " << triplets[0].S1.D << "   " << triplets[0].S2.D << std::endl;
	 std::cout << "# Es " << triplets[0].S1.E << "   " << triplets[0].S2.E << std::endl;
	 std::cout << "# B " << triplets[0].S1.B << "   " << triplets[0].S2.B << std::endl;
	 std::cout << "# M1 " << triplets[0].S1.rot.matrix()  << std::endl;
	 std::cout << "# M2 " << triplets[0].S2.rot.matrix()  << std::endl;
	 std::cout << "# J " << triplets[0].J << std::endl;
	 std::cout << "# Jdip " << triplets[0].Jdip << std::endl;
	 std::cout << "# r12 " << triplets[0].r12 << std::endl;
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
     std::cout << "# Jmax " <<  Jmax << std::endl;
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


const int Triplet_Pair_From_Gene::NPAIRS = 1;

static const int NVARS = Triplet_Pair_From_Gene::N_TP * Triplet_Pair_From_Gene::NPAIRS;

struct genotype
{
  double gene[NVARS];
  double score;
  double fitness;
  double upper[NVARS];
  double lower[NVARS];
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
static const int POPSIZE = 30;

template <class fitness_finder> class simple_GA { 
   fitness_finder &ffinder;
    
   int int_uniform_ab ( int a, int b ) { 
      return a + (rand() % (b - a + 1));
   }

   double real_uniform_ab ( double a, double b ) { 
      return a + (b - a) * (double) rand() / (double) RAND_MAX;
   }

   void Xover ( int one, int two ) {
      //  Select the crossover point.
      int point = int_uniform_ab ( 0, NVARS - 1 );
      //  Swap genes in positions 0 through POINT-1.
      for (int i = 0; i < point; i++ ) {
	 double t = population[one].gene[i];
	 population[one].gene[i] = population[two].gene[i];
	 population[two].gene[i] = t;
      }
   }

   void copy_gene(int from, int to) { 
      for (int i = 0; i < NVARS; i++ ) {
        population[to].gene[i] = population[from].gene[i];
      }
      population[to].score = population[from].score;
      population[to].fitness = population[from].fitness;      
   }

public : 
   double PXOVER = 0.8;
   double PMUTATION = 0.1;
   struct genotype population[POPSIZE+1];
   struct genotype newpopulation[POPSIZE+1]; 
   double temp;

   simple_GA(fitness_finder &f) : ffinder(f) {
      PXOVER = 0.8;
      PMUTATION = 0.1;
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
     
     best = worst = population[0].fitness;
     best_mem = worst_mem = 0;

     for (i = 0; i < POPSIZE - 1; ++i) {
        if ( population[i+1].fitness < population[i].fitness ) {
	   if ( best <= population[i].fitness ) {
	      best = population[i].fitness;
	      best_mem = i;
	   }
	   
	   if ( population[i+1].fitness <= worst ) {
	      worst = population[i+1].fitness;
	      worst_mem = i + 1;
	   }
	} else {
	  if ( population[i].fitness <= worst ) {
	     worst = population[i].fitness;
	     worst_mem = i;
	  }
	  if ( best <= population[i+1].fitness ) {
	     best = population[i+1].fitness;
	     best_mem = i + 1;
	  }
	}
     }

     if ( population[POPSIZE].fitness <= best ) {
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
	 population[member].score = ffinder.score(population[member].gene);
      }
      double avscore = 0.0;
      for (int member = 0; member < POPSIZE; member++ ) { 
	 avscore += population[member].score/(double) (POPSIZE+1);
      }
      for (int member = 0; member < POPSIZE; member++ ) { 
	// towards minimum
	// 	 population[member].fitness = exp ( -(population[member].score - avscore)/temp );
	// towards maximum
	//	population[member].fitness = exp ( (population[member].score - avscore)/temp );
	 population[member].fitness = population[member].score;
      }
   }

  /**
   void initialize (const double *init = NULL) {
      for (int j = 0; j <= POPSIZE; j++ ) {
         for (int i = 0; i < NVARS; i++ ) {
	    population[j].fitness = 0;	
	    population[j].score = 0;	
	    population[j].rfitness = 0;
	    population[j].cfitness = 0;
	    population[j].lower[i] = ffinder.lower(i);
	    population[j].upper[i] = ffinder.upper(i);
	    if (init == NULL) { 
	       if (j) { population[j].gene[i] = real_uniform_ab (population[j].lower[i], population[j].upper[i]); }
	       else { population[j].gene[i] = (population[j].lower[i] + population[j].upper[i]) / 2.0; }	      
	    } else { 
	       population[j].gene[i] = init[i];
	    }
	 }
      }
   }  
  **/

   void initialize (const double *init = NULL) {
      for (int j = 0; j <= POPSIZE; j++ ) {
         for (int i = 0; i < NVARS; i++ ) {
	    population[j].fitness = 0;	
	    population[j].score = 0;	
	    population[j].rfitness = 0;
	    population[j].cfitness = 0;
	    population[j].lower[i] = ffinder.lower(i);
	    population[j].upper[i] = ffinder.upper(i);
	    population[j].gene[i] = real_uniform_ab (population[j].lower[i], population[j].upper[i]); 
	 }
      }
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
	 for (int j = 0; j < NVARS; j++ ) {
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
	 for (int j = 0; j < NVARS; j++ ) {
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

   void print_info(void) { 
      std::cout << "# POPSIXE " << POPSIZE << std::endl;
      std::cout << "# NVARS " << NVARS << std::endl;
      std::cout << "# PXOVER " << PXOVER << std::endl;
      std::cout << "# PMUTATION " << PMUTATION << std::endl;
      std::cout << "# temp " << temp << std::endl;
   }

   void print_best(int generation) { 
      std::cout << "# best gene = " << population[POPSIZE].fitness << "\n";
      for (int i = 0; i < NVARS; i++ ) {
         std::cout << generation << "   " << i << "    " << population[POPSIZE].gene[i] << "  %" << std::endl;
      }
      std::cout << "# with fitness = " << population[POPSIZE].fitness << "\n";
      //      ffinder.print_gene(population[POPSIZE].gene);
   }
};

main()
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
    odmr.Jmax = odmr.Dmax;
    odmr.Jdip_max = odmr.Dmax;
    odmr.Amp_max = 100.0;
    odmr.Gamma_max = 50.0;
    odmr.use_chi1 = true;
    odmr.print_info();

    /***
    double gene0[NVARS];
    odmr.read_gene("opt2.gene", gene0);
    odmr.print_gene(gene0);
    exit(0);
    ***/

    simple_GA<Triplet_Pair_From_Gene> ga(odmr);
    ga.initialize();
    ga.print_info();
    ga.evaluate ();
    std::cout << "# ga evaluate "  << std::endl;
    ga.report(-1);
    ga.keep_the_best();

    int MAXGENS = 100000;

    for (int generation = 0; generation < MAXGENS; generation++ ) {
      //       ga.temp *= (1.0 - anneal_eps);
      //       if (ga.temp < temp_min) ga.temp = temp_min;
      //       tex.nav *= (1.0 + anneal_eps);
      //       if (tex.nav > nav_max) { tex.nav = nav_max; }
       ga.selector ();
       ga.crossover ();
       ga.mutate ();
       ga.report(generation);
       ga.evaluate ();
       ga.elitist();
       if (!(generation % 20)) {
	  ga.print_best(generation);	  
       }
    }
    ga.print_best(MAXGENS);
    
}
