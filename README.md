# codes for ODMR analysis, python and C++ versions 

odmr_triplets.cc -> C++ 

C++ class organization 

class Rotation -> deals with 3D space rotations 

Pauli Matrices - defines spin matrices 
struct PauliTripletMatrices [3x3]
struct PauliMatrices [2x2]

GenericSpinBase - virtual functions defined for all spin types 
   |
   GenericSpin - template implementation of various spins 

typedef GenericSpin<PauliTripletMatrices> TripletSpin;
typedef GenericSpin<PauliMatrices> SpinHalf;

template <class Spin1, class Spin2> class SpinPair - template implementation of a pair of spins 

template <typename... Tp> struct SpinTuple - template implementation of an arbitrarty number of coupled spins 

class TripletPair : public SpinPair<TripletSpin, TripletSpin> - additional functions for a pair of triplets 

template <class SpinSystem>  class ODMR_Signal - generic framework to compute magnetic resonance response 

template <class SpinSystem>  class Merrifield - generic framework to compute fluorescence versus magnetic field

---------------------------

triplet.py -> python version somewhat lagging behind in coding 
