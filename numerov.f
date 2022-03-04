C      NUMEROV METHOD TO SOLVE THE  SCHRODINGER EQUATION
C      CASE:  H-ATOM
C------------------------------------------------------------------------------------
       IMPLICIT REAL *8(A-H,O-Z)
       PARAMETER (N=1000000)
C      F: WAVE FUNCTION,WHERE F(r)=r.R(r)
C      R: RADIAL DISTANCE
C      THE NEEDED FORM TO NUMERICALLY SOLVE THE RADIAL SCHRODINGER EQUATION IS: F"(r)=G(r).F(r)
C      WHERE G: G(r)=(2V-2E)+ (L(L+1)/r**2), WHERE V =(-1/r)
C      ALL THE CALCULATIONs HEVE BEEN PERFORMED IN ATOMIC UNIT(a.u.)
C............................................................................................................................
       DIMENSION F(N),R(N),G(N)
!      M: NUMBER OF GRID POINTS
       M=30000
       RMIN = 1E-15
       RMAX = 100.0
       H=(RMAX-RMIN)/M
       SS=H*H/12.0

       WRITE(*,*)'GIVE THE VALUE OF PRINCIPLE QUANTUM NUMBER'
       READ(*,*) NQ
       WRITE(*,*)'GIVE THE VALUE OF AZIMUTHAL QUANTUM NUMBER'
       READ(*,*) L

       F(1)=0.0    ! INITIAL VALUE OF WAVE FUNCTION (APPROXIMATED TO ZERO)
       F(2)=0.000001
       R(1)=RMIN
       R(2)=R(1)+ H

C      ENERGY RANGE WITHIN WHICH WE WANT TO SEARCH FOR EIGEN VALUES
       ELOW= -0.60
       EHIGH= -0.002
        DO  K=1,M
        E=(EHIGH+ELOW)/2.0
        NODE=0
C       GENERATING THE WAVE FUNCTIONS...
        G(1)=((-2.0/R(1))-2.0*E)+ ((L*(L+1)/(R(1)**2)))
        G(2)=((-2.0/R(2))-2.0*E)+ ((L*(L+1)/(R(2)**2)))
        DO  I=2,M-1
        R(I+1)=R(I)+ H
        G(I+1)=((-2.0/R(I+1))-2.0*E)+ ((L*(L+1)/(R(I+1)**2)))
        F(I+1)=2.0D0*F(I)-F(I-1)+ 10.0*G(I)*F(I)*SS + G(I-1)*F(I-1)*SS
        F(I+1)= F(I+1)/(1.0 - G(I+1)*SS)
C       CHECKING FOR NODE. IF NODE IS PRESENT
        IF(F(I+1)*F(I).LT. 0.0)THEN
        NODE = NODE + 1
        ENDIF
        ENDDO
C       CONDITION TO REACH CLOSER TO THE EIGEN VALUE
        IF (NODE .GT. NQ-L-1)THEN
        EHIGH=E
        ENDIF
        IF (NODE .LE.  NQ-L-1)THEN
        ELOW=E
        ENDIF
        ENDDO

  5     FORMAT(5(4X,I3,6X,F10.7))
        WRITE(*,*)'   NODES    ENERGY (a.u)'
C       WRITING THE NUMBER OF NODES AND ENERGY
        WRITE(*,5)  NQ-L-1 ,E
        STOP
        END

