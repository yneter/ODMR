# codes for ODMR analysis, python and C++ versions 

triplet.py -> python 
odmr_triplets.cc -> C++ 
implementation is very similar 
 
class TripletHamiltonian -> grouping all functions that deal with a single triplet

class TwoTriplets -> with all functions dealing with two triplets which uses TripletHamiltonian through composition and feating a TripletHamiltonian triplet instance

class ODMR_Signal -> generic framework to compute magnetic resonance response, an use TwoTriplets to compute MR/ODMR signals 

