      SUBROUTINE wy_hess(SIJT, MO, EIA, NOCC, NVIR, NPOT, HESS)
!*  =====================================================================
!     GET CUBIC SPLINE COEFFICIENT FROM X, Y
!     X : VECTOR SIZE N
!     Y : VECTOR SIZE N
!     N : VECTOR SIZE
!     RETURN: CUBIC_COEFF(4,N-1)
!     SCIPY.CUBICSPLINE
!*  =====================================================================
      USE omp_lib
      IMPLICIT NONE
!     Scalar Arguments
      INTEGER NOCC, NVIR, NPOT, NBAS
      INTEGER I, A, U, T
!     Array Arguments
      DOUBLE PRECISION MO(NOCC+NVIR,NOCC+NVIR)
      DOUBLE PRECISION SIJT(NOCC+NVIR, NOCC+NVIR, NPOT)
      DOUBLE PRECISION EIA(NOCC, NVIR)
      DOUBLE PRECISION SIAT(NOCC, NVIR, NPOT)
      DOUBLE PRECISION STMP(NOCC, NOCC+NVIR, NPOT)
      EXTERNAL DGEMM
!     Output Arguments
      DOUBLE PRECISION, INTENT(OUT) :: HESS(NPOT, NPOT)
!*  =====================================================================

      NBAS = NOCC + NVIR

!$OMP PARALLEL DO PRIVATE(STMP)
      DO T=1,NPOT
        CALL DGEMM('T','N',NOCC,NBAS,NBAS,1.D0,MO(:,:NOCC), &
                   NBAS,SIJT(:,:,T),NBAS,0.D0,STMP(:,:,T),NOCC)
        CALL DGEMM('N','N',NOCC,NVIR,NBAS,1.D0,STMP(:,:,T), &
                   NOCC,MO(:,NOCC+1:),NBAS,0.D0,SIAT(:,:,T),NOCC)
      END DO
!$OMP END PARALLEL DO

      !$OMP PARALLEL DO REDUCTION(+:HESS)
      DO T=1,NBAS
          DO U=1,NBAS
              DO A=1,NVIR
                  DO I=1,NOCC
                      HESS(U,T) = HESS(U,T) &
                                 + SIAT(I,A,U) * SIAT(I,A,T) / EIA(I,A)
                  END DO
           END DO
          END DO
      END DO
!$OMP END PARALLEL DO

      HESS = 4.D0 * HESS

      RETURN
      END SUBROUTINE

      SUBROUTINE einsum_ij_ijt_2t( A, B, N1, N2, C )
!*  =====================================================================
!     CALCULATE C = np.einsum('ij,ijt->t',A,B)
!     A : ARRAY SHAPE (N1,N1)
!     B : ARRAY SHAPE (N1,N1,N2)
!     ASSUME A(ij)=A(ji), B(ijt)=B(jit)
!*  =====================================================================
      USE omp_lib
      IMPLICIT NONE
!     Scalar Arguments 
      INTEGER N1,N2
      INTEGER I,J,T
!     Array Arguments 
      DOUBLE PRECISION  A(N1,N1)
      DOUBLE PRECISION  B(N1,N1,N2)
!     Output Arguments
      DOUBLE PRECISION,INTENT(OUT) :: C(N2)
!*  =====================================================================

!$OMP PARALLEL DO REDUCTION(+:C)
      DO T=1,N2
          DO J=1,N1
              DO I=J+1,N1
                  C(T)=C(T)+2.D0*A(I,J)*B(I,J,T)
              END DO
          END DO
      END DO
!$OMP END PARALLEL DO

!$OMP PARALLEL DO
      DO T=1,N2
          DO I=1,N1
               C(T)=C(T)+A(I,I)*B(I,I,T)
          END DO
      END DO
!$OMP END PARALLEL DO

      RETURN
      END SUBROUTINE

      SUBROUTINE ovlp_aab( WEIGHT, AO1, AO2, N1, N2, NG, SIJT )
!*  =====================================================================
!     CALCULATE THREE-CENTER OVERLAP INTEGRAL NUMERICALLY
!     WEIGHT : VECTOR SIZE NG
!     AO1 : MATRIX SHAPE (NG, N1)
!     AO2 : MATRIX SHAPE (NG, N2)
!     RETURN: THREE-CENTER OVERLAP INTEGRAL SIJT
!*  =====================================================================
      USE omp_lib
      IMPLICIT NONE
!     Scalar Arguments
      INTEGER N1, N2, NG, I, J, K, R
!     Array Arguments
      DOUBLE PRECISION,INTENT(IN) :: AO1(NG,N1)
      DOUBLE PRECISION,INTENT(IN) :: AO2(NG,N2)
      DOUBLE PRECISION,INTENT(IN) :: WEIGHT(NG)
!     Output Arguments
      DOUBLE PRECISION,INTENT(OUT) :: SIJT(N1,N1,N2)
!*  =====================================================================

!$OMP PARALLEL DO REDUCTION(+:SIJT)
      DO I=1,N2
          DO J=1,N1
              DO K=J,N1
                  DO R=1,NG
                      SIJT(K,J,I)=SIJT(K,J,I)+AO1(R,K)*AO1(R,J)*AO2(R,I)*WEIGHT(R)
                  END DO
              END DO
          END DO
      END DO
!$OMP END PARALLEL DO

      DO I=1,N2
          DO J=1,N1
              DO K=J+1,N1
                  SIJT(J,K,I)=SIJT(K,J,I)
              END DO
          END DO
      END DO      

      RETURN
      END SUBROUTINE

      SUBROUTINE eval_vhm( C, ZVH, COORDS, RAD, LMAX, NG, N_RAD, VH )
!*  =====================================================================
!TO SUM UP PIECEWISE HARTREE POTENTIAL TO TOTAL HARTREE POTENTIAL
!*  =====================================================================
      USE omp_lib
      IMPLICIT NONE
!     Scalar Arguments
      INTEGER            J, X, Y, NG, LMAX, N_RAD
      DOUBLE PRECISION   PI
!     Array Arguments
      INTEGER            L( (LMAX+1)*(LMAX+1) )
      DOUBLE PRECISION   C( N_RAD,(LMAX+1)*(LMAX+1) ), C1(4, N_RAD-1)
      DOUBLE PRECISION   C2(4, N_RAD-1), CUBIC_COEFF(4, N_RAD-1)
      DOUBLE PRECISION   RAD(N_RAD), COORDS(NG)
      DOUBLE PRECISION   V(NG), I(N_RAD), I1(N_RAD), I2(N_RAD)
      DOUBLE PRECISION   ZVH(NG,(LMAX+1)*(LMAX+1))
!     Output Arguments
      DOUBLE PRECISION, INTENT(OUT) :: VH(NG)
!*  =====================================================================

      PI = 4.D0*ATAN(1.D0)

!$OMP PARALLEL DO
      DO X=0,LMAX
          DO Y=-X,X
              L(X*(X+1)+Y+1) = X
          END DO
      END DO
!$OMP END PARALLEL DO

!$OMP PARALLEL DO PRIVATE(V, C2, I, I2, I1, CUBIC_COEFF, C1) REDUCTION(+:VH)
      DO J=1,(LMAX+1)**2
          CALL cubicspline( RAD, C(:,J), N_RAD, C1 )
          CALL convert( C1, RAD, N_RAD, CUBIC_COEFF )
          CALL eval_I1( CUBIC_COEFF, RAD, L(J), N_RAD, I1 )
          CALL eval_I2( CUBIC_COEFF, RAD, L(J), N_RAD, I2 )
          I = I1 / ( RAD**( L(J)+1.D0) ) + I2 * ( RAD**( L(J) ) )
          CALL cubicspline( RAD, I, N_RAD, C2 )
          CALL eval_cubicspline( COORDS, RAD, C2, NG, N_RAD, V )
          VH = VH + ZVH(:,J)*V / (2.D0*L(J) + 1.D0)
      END DO
!$OMP END PARALLEL DO

      VH = 4.D0 * PI * VH

      RETURN
      END SUBROUTINE

      SUBROUTINE cubicspline( X, Y, N, CUBIC_COEFF )
!*  =====================================================================
!     GET CUBIC SPLINE COEFFICIENT FROM X, Y
!     X : VECTOR SIZE N
!     Y : VECTOR SIZE N
!     N : VECTOR SIZE
!     RETURN: CUBIC_COEFF(4,N-1)
!     SCIPY.CUBICSPLINE
!*  =====================================================================
      IMPLICIT NONE
!     Scalar Arguments
      INTEGER            N, J, INFO
      DOUBLE PRECISION   D
!     Array Arguments
      INTEGER            IPIV(N)
      DOUBLE PRECISION   A(4,N), B(N), X(N), Y(N), S(N-1), T(N-1), DX(N-1)
!     Output Arguments
      DOUBLE PRECISION,INTENT(OUT) :: CUBIC_COEFF(4,N-1)
!*  =====================================================================
      EXTERNAL dgbsv
!*  =====================================================================

      DX = X(2:) - X(:N-1)
      S = (Y(2:)-Y(:N-1)) / DX

      A(2,3:) = DX(:N-2)                      !A(2) THE UPPER DIAGONAL
      A(3,2:N-1) = 2.D0 * (DX(:N-2) + DX(2:)) !A(3) THE DIAGONAL
      A(4,:N-2) = DX(2:)                      !A(4) THE LOWER DIAGONAL
      B(2:N-1) = 3.D0 * (DX(2:)*S(:N-2)+DX(:N-2)*S(2:))

      !BOUNDARY CONDITION 'NOT-A-KNOT'
      !START POINT
      A(2,2) = X(3) - X(1)                    !A(2) THE UPPER DIAGONAL
      A(3,1) = DX(2)                          !A(3) THE DIAGONAL
      D = X(3) - X(1)
      B(1) = ((DX(1)+2.D0*D)*DX(2)*S(1)+DX(1)**2.D0*S(2))/ D

      !END POINT
      A(3,N) = DX(N-2)                        !A(3) THE DIAGONAL
      A(4,N-1) = X(N) - X(N-2)                !A(4) THE LOWER DIAGONAL
      D = X(N) - X(N-2)
      B(N) = ((DX(N-1)+2.D0*D)*DX(N-2)*S(N-1)+DX(N-1)**2.D0*S(N-2))/ D

      !COMPUTE COEFFICIENTS
      CALL dgbsv(N,1,1,1,A,4,IPIV,B,N,INFO)
      !SUB-DIAGONAL, DIAGONAL, SUPER-DIAGONAL

      T =( B(:N-1) + B(2:) - 2.D0 * S ) / DX

      CUBIC_COEFF(1,:) = T / DX
      CUBIC_COEFF(2,:) = ( S - B(:N-1) ) / DX - T
      CUBIC_COEFF(3,:) = B(:N-1)
      CUBIC_COEFF(4,:) = Y(:N-1)

      RETURN
      END SUBROUTINE

      SUBROUTINE convert(C,RAD,N_RAD,C_OUT)
!*  =====================================================================
      IMPLICIT NONE
!     Scalar Arguments
      INTEGER            N_RAD
!     Array Arguements
      DOUBLE PRECISION   C(4,N_RAD-1), RAD(N_RAD)
!     Output Arguments
      DOUBLE PRECISION, INTENT(OUT) :: C_OUT(4,N_RAD-1)
!*  =====================================================================

      C_OUT(1,:) = C(1,:)
      C_OUT(2,:) = C(2,:) -3.D0*C(1,:)*RAD(:N_RAD-1)
      C_OUT(3,:) = C(3,:) -2.D0*C(2,:)*RAD(:N_RAD-1) &
                   + 3.D0*C(1,:)*RAD(:N_RAD-1)**2.D0
      C_OUT(4,:) = C(4,:) -1.D0*C(3,:)*RAD(:N_RAD-1) &
                   +1.D0*C(2,:)*RAD(:N_RAD-1)**2.D0  &
                   -1.D0*C(1,:)*RAD(:N_RAD-1)**3.D0

      RETURN
      END SUBROUTINE

      SUBROUTINE eval_I1( CUBIC_COEFF, RAD, L, N_RAD, I1 )
!*  =====================================================================
      IMPLICIT NONE
!     Scalar Arguments
      INTEGER            I, J, L, N_RAD
      DOUBLE PRECISION   DDOT
!     Array Arguments
      DOUBLE PRECISION   CUBIC_COEFF(4,N_RAD-1), RAD(N_RAD)
      DOUBLE PRECISION   INTEGRAND(N_RAD-1,4), TMP(N_RAD-1)
!     Output Arguments
      DOUBLE PRECISION, INTENT(OUT) :: I1(N_RAD)
!*  =====================================================================

      I1 = 0.D0

      DO J=0,3
          INTEGRAND(:,J+1) = (RAD(2:)**(6.D0+L-J) &
                             - RAD(:N_RAD-1)**(6.D0+L-J)) / (6.D0+L-J)
      END DO

      DO I=1,N_RAD-1
          TMP(I) = DDOT(4,INTEGRAND(I,:),1,CUBIC_COEFF(:,I),1)
      END DO

      DO I=2,N_RAD
          I1(I) = I1(I-1) + TMP(I-1)
      END DO

      RETURN
      END SUBROUTINE

      SUBROUTINE eval_I2( CUBIC_COEFF, RAD, L, N_RAD, I2 )
!*  =====================================================================
      IMPLICIT NONE
!     Scalar Arguments
      INTEGER            I, J, L, N_RAD
      DOUBLE PRECISION   DDOT
!     Array Arguements
      DOUBLE PRECISION   CUBIC_COEFF(4,N_RAD-1), RAD(N_RAD)
      DOUBLE PRECISION   INTEGRAND(N_RAD-1,4), TMP(N_RAD-1)
!     Output Arguments
      DOUBLE PRECISION, INTENT(OUT) :: I2(N_RAD)
!*  =====================================================================

      I2 = 0.D0

      IF (L .LT. 2 .OR. L .GT. 5) THEN
          DO J=0,3
              INTEGRAND(:,J+1) = (RAD(2:)**(5.D0-L-J) &
                                 - RAD(:N_RAD-1)**(5.D0-L-J)) / (5.D0-L-J)
          END DO
      ELSEIF (L .EQ. 2) THEN
          DO J=0,2
              INTEGRAND(:,J+1) = (RAD(2:)**(5.D0-L-J) &
                                 - RAD(:N_RAD-1)**(5.D0-L-J)) / (5.D0-L-J)
          END DO
          INTEGRAND(:,4) = LOG(RAD(2:)/RAD(:N_RAD-1))
      ELSEIF (L .EQ. 3) THEN
          DO J=0,3
              INTEGRAND(:,J+1) = (RAD(2:)**(5.D0-L-J) &
                                 - RAD(:N_RAD-1)**(5.D0-L-J)) /(5.D0-L-J)
          END DO
              INTEGRAND(:,3) = LOG(RAD(2:)/RAD(:N_RAD-1))
      ELSEIF (L .EQ. 4) THEN
          DO J=0,3
              INTEGRAND(:,J+1) = (RAD(2:)**(5.D0-L-J) &
                                 - RAD(:N_RAD-1)**(5.D0-L-J))/(5.D0-L-J)
          END DO
          INTEGRAND(:,2) = LOG(RAD(2:)/RAD(:N_RAD-1))
      ELSEIF (L .EQ. 5) THEN
          DO J=1,3
              INTEGRAND(:,J+1) = (RAD(2:)**(5.D0-L-J) &
                                 - RAD(:N_RAD-1)**(5.D0-L-J))/(5.D0-L-J)
          END DO
              INTEGRAND(:,1) = LOG(RAD(2:)/RAD(:N_RAD-1))
      ENDIF

      DO I=1,N_RAD-1
          TMP(I) = DDOT(4,INTEGRAND(I,:),1,CUBIC_COEFF(:,I),1)
      END DO

      DO I=2,N_RAD
          I2(N_RAD-I+1) = I2(N_RAD-I+2) + TMP(N_RAD-I+1)
      END DO

      RETURN
      END SUBROUTINE

      SUBROUTINE eval_cubicspline( X, RLIST, C, NG, N_RAD, S )
!*  =====================================================================
!     EVALUATE THE VALUE OF CUBIC SPLINE FUNCTION WITH GIVEN CUBIC_COEFF
!*  =====================================================================
      IMPLICIT NONE
!     Scalar Arguments
      INTEGER           R, J, N_RAD, NG
!     Array Arguments
      DOUBLE PRECISION  X(NG), C(4,N_RAD-1), RLIST(N_RAD)
!     Output Arguments
      DOUBLE PRECISION, INTENT(OUT) :: S(NG)
!*  =====================================================================

      J = 1
      R = 1

      DO WHILE ((R .LE. NG) .AND. (J .LE. N_RAD) )
          IF ( ( RLIST(J) .LE. X(R) ) .AND. ( X(R) .LT. RLIST(J+1)) ) THEN
              S(R) = C(4,J) + C(3,J) * ( X(R) - RLIST(J) )**(1) &
                     + C(2,J) * ( X(R) - RLIST(J) )**(2) &
                     + C(1,J) * ( X(R) - RLIST(J) )**(3)
              R = R + 1
          ELSE IF (X(R) .LE. RLIST(1)) THEN
              S(R) = C(4,J) + C(3,J) * ( X(R) - RLIST(J) )**(1) &
                     + C(2,J) * ( X(R) - RLIST(J) )**(2) &
                     + C(1,J) * ( X(R) - RLIST(J) )**(3)
              R = R + 1
          ELSE IF (X(R) .GE. RLIST(N_RAD)) THEN
              S(R) = C(4,J) + C(3,J) * ( X(R) - RLIST(J) )**(1) &
                     + C(2,J) * ( X(R) - RLIST(J) )**(2) &
                     + C(1,J) * ( X(R) - RLIST(J) )**(3)
              R = R + 1
          ELSE
              J = J + 1
          END IF
      END DO

      RETURN
      END SUBROUTINE


