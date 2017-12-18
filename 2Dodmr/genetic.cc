

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
   enum { NVARS = fitness_finder::NVARS };

    
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

   bool is_fixed[NVARS];
   double fixed_value[NVARS];

public : 
   enum { POPSIZE = 16 };


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
      for (int var = 0; var < NVARS; var++) { 
	 is_fixed[var] = false;
      }
   }

   void mutate (void) { 
      const double a = 0.0;
      const double b = 1.0;
      double lbound;
      double ubound;
      double x;

      for (int i = 0; i < POPSIZE; i++ ) {
	 for (int j = 0; j < NVARS; j++ ) {
	    if ( is_fixed[j] ) { 
	       population[i].gene[j] = fixed_value[j];
	    } else { 
	       x = real_uniform_ab (a, b);
	       if (x < PMUTATION ) {	      
		  lbound = population[i].lower[j];
		  ubound = population[i].upper[j];
		  population[i].gene[j] = real_uniform_ab (lbound, ubound);
	       } 
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
	    if ( is_fixed[j] ) { 
	       population[i].gene[j] = fixed_value[j];
	    } else { 
	       x = real_uniform_ab (a, b);
	       if ( x < PMUTATION ) {	      
		  lbound = std::max(population[i].lower[j], population[i].gene[j] - amplitude);
		  ubound = std::min(population[i].upper[j], population[i].gene[j] + amplitude);
		  population[i].gene[j] = real_uniform_ab (lbound, ubound);
	       }
	    }
	 }
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

   void fix(int var, double value) { 
     if (var >= 0 && var < NVARS) {
        is_fixed[var] = true;
	fixed_value[var] = value;
     }
   }

   void fix_to_gene(double gene[]) { 
      for (int i = 0; i < NVARS; i++ ) {
         is_fixed[i] = true;
         fixed_value[i] = gene[i];
      }
   }

   void unfix(int var) { 
     if (var >= 0 && var < NVARS) {
        is_fixed[var] = false;
     }
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
         for (int i = 0; i < NVARS; i++ ) {
	    population[j].lower[i] = ffinder.lower(i);
	    population[j].upper[i] = ffinder.upper(i);
	    if (is_fixed[i]) { 
	       population[j].gene[i] = fixed_value[i]; 	       
	    } else { 
	       population[j].gene[i] = real_uniform_ab (population[j].lower[i], population[j].upper[i]); 
	    }
	 }
      }
   }  

   void initial_values(int num, const double *gene) { 
      std::cout << "# setting initial_value for gene " << num << std::endl;
      for (int i = 0; i < NVARS; i++ ) {
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
      stddev = sqrt ( ( sum_square - square_sum ) / (double) POPSIZE );
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
   }
};
