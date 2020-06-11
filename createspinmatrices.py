
import math
import sys


def matrix_name(N) : 
    if (N % 2) == 0:
        S = "%dp5" % ((N-2)/2)
    else:
        S = "%d" % ( (N-1)/2 )
    return ("Spin%s" % S)

def print_Sz_matrix(N) : 
    if (N % 2) == 0:
        S = "%d/%d" % (N-1, 2)
    else:
        S = "%d" % ( (N-1)/2 )
    matrixstring = matrix_name(N)
    sys.stdout.write("const %s::SpinMatrix %s::Sz ( ( (SpinMatrix() << " % (matrixstring, matrixstring))
    for n in range(N) : 
        for m in range(N):
            if (m != n):
                sys.stdout.write(" 0")
            else:
                if (N % 2) != 0: 
                    Sz = int((N-1)/2 - n)
                    sys.stdout.write(" %s" % Sz)                
                else:
                    Sz2 = int((N-1) - 2*n)
                    sys.stdout.write(" %s./2." % Sz2)                

            if (m != N-1) or (n != N-1):
                sys.stdout.write(",")

    print(").finished() ) );")



def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def print_Sxy_matrix(N, Sx = True) : 
    if (N % 2) == 0:
        S = "%d/%d" % (N-1, 2)
    else:
        S = "%d" % ( (N-1)/2 )
    matrixstring = matrix_name(N)
    if Sx:
        sys.stdout.write("const %s::SpinMatrix %s::Sx ( ( (SpinMatrix() << " % (matrixstring, matrixstring))
    else :
        sys.stdout.write("const %s::SpinMatrix %s::Sy ( ( (SpinMatrix() << " % (matrixstring, matrixstring))

    for n in range(N) : 
        for m in range(N):
            if (m - n != -1 and m - n != 1):
                sys.stdout.write(" 0")
            else:
                if (N % 2) != 0: 
                    # even spin 
                    S = int((N-1)/2)
                    Mz = int(S - m)
                    Nz = int(S - n)
                    K = int(S*(S+1)-Mz*Nz)
                    if K != 0:
                        if Sx == False:
                            if (m + 1 == n): 
                                sys.stdout.write(" -iii *")
                            else: 
                                sys.stdout.write(" iii *")                                
                        sk = isqrt(K)
                        if sk*sk != K: 
                            if K % 2 != 0:
                                sys.stdout.write(" sqrt(%d.)/2." % (K))
                            else:
                                sys.stdout.write(" sqrt(%d./2.)" % (int(K/2)))                                
                        else:
                            if (sk % 2) != 0:
                                sys.stdout.write(" %d./2." % (sk))
                            else:
                                sys.stdout.write(" %d." % (sk/2))
                    else:
                        sys.stdout.write(" 0.")
                else:
                    # odd spin 
                    S2 = int(N-1)
                    Mz2 = int(S2 - 2*m)
                    Nz2 = int(S2 - 2*n)
                    K4 = int(S2*(S2+2)-Mz2*Nz2)
                    if K4 != 0:
                        if Sx == False:
                            if (m + 1 == n): 
                                sys.stdout.write(" -iii *")
                            else: 
                                sys.stdout.write(" iii *")                                
                        sk2 = isqrt(K4)
                        if sk2*sk2 != K4: 
                            if (K4 % 4) == 0:
                                sys.stdout.write(" sqrt(%d.)/2." % (K4/4))
                            else : 
                                sys.stdout.write(" sqrt(%d.)/4." % (K4))
                            # sys.stdout.write(" sqrt(%d.)/4." % (K4))
                        else:
                            if (sk2 % 4) == 0:
                                sys.stdout.write(" %d." % (sk2/4))
                            elif (sk2 % 2) == 0:
                                sys.stdout.write(" %d./2." % (sk2/2))
                            else : 
                                sys.stdout.write(" %d./4." % (sk2))
                            
                    else:
                        sys.stdout.write(" 0.")

            if (m != N-1) or (n != N-1):
                sys.stdout.write(",")

    print(").finished() ) );")

def print_Id(N) : 
    matrixstring = matrix_name(N)
    print("const %s::SpinMatrix %s::Id (  (SpinMatrix() = SpinMatrix::Identity()) );" % (matrixstring,matrixstring) )

def print_SpSm(N) : 
    matrixstring = matrix_name(N)
    print("const %s::SpinMatrix %s::Sp (  (SpinMatrix() = %s::Sx + iii * %s::Sy) );" % (matrixstring,matrixstring,matrixstring,matrixstring) )
    print("const %s::SpinMatrix %s::Sm (  (SpinMatrix() = %s::Sx - iii * %s::Sy) );" % (matrixstring,matrixstring,matrixstring,matrixstring) )

def print_header(N):
    print("""struct %s { 
    enum { matrix_size = %d };
    typedef Matrix<complexg, matrix_size, matrix_size>  SpinMatrix;
    static const SpinMatrix Sx;
    static const SpinMatrix Sy;
    static const SpinMatrix Sz;
    static const SpinMatrix Id;
    static const SpinMatrix Sp;
    static const SpinMatrix Sm;
};""" % (matrix_name(N), N))

Nmax = 12
for N in range(2, Nmax):
    print_header(N)
    print_Sxy_matrix(N, Sx=True)
    print_Sxy_matrix(N, Sx=False)
    print_Sz_matrix(N)
    print_Id(N)
    print_SpSm(N)
    print("")


for N in range(2, Nmax): 
    matrixstring = matrix_name(N)
    print("typedef SingleSpin<%s> Single%s;" % (matrixstring, matrixstring))


