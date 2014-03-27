#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define TRUE 1

/***** Globals ******/
float **a; /* The coefficients */
float *x;  /* The unknowns */
float *xnew;  /* newly calculated unknowns */
float *b;  /* The constants */
float err; /* The absolute relative error */
float *rerr;  /* array of relative errors */
int num = 0;  /* number of unknowns */


/****** Function declarations */
void check_matrix(); /* Check whether the matrix will converge */
void get_input();  /* Read input from file */
void jacobi(); /* Recalculate x */

/********************************/



/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/

/* 
   Conditions for convergence (diagonal dominance):
   1. diagonal element >= sum of all other elements of the row
   2. At least one diagonal element > sum of all other elements of the row
 */
void check_matrix()
{
  int bigger = 0; /* Set to 1 if at least one diag element > sum  */
  int i, j;
  float sum = 0;
  float aii = 0;
  
  for(i = 0; i < num; i++)
  {
    sum = 0;
    aii = fabs(a[i][i]);
    
    for(j = 0; j < num; j++)
       if( j != i)
   sum += fabs(a[i][j]);
       
    if( aii < sum)
    {
      printf("The matrix will not converge\n");
      exit(1);
    }
    
    if(aii > sum)
      bigger++;
    
  }
  
  if( !bigger )
  {
     printf("The matrix will not converge\n");
     exit(1);
  }
}


/******************************************************/
/* Read input from file */
void get_input(char filename[])
{
  FILE * fp;
  int i,j;  
 
  fp = fopen(filename, "r");
  if(!fp)
  {
    printf("Cannot open file %s\n", filename);
    exit(1);
  }

 fscanf(fp,"%d ",&num);
 fscanf(fp,"%f ",&err);

 /* Now, time to allocate the matrices and vectors */
 a = (float**)malloc(num * sizeof(float*));
 if( !a)
  {
  printf("Cannot allocate a!\n");
  exit(1);
  }

 for(i = 0; i < num; i++) 
  {
    a[i] = (float *)malloc(num * sizeof(float)); 
    if( !a[i])
    {
    printf("Cannot allocate a[%d]!\n",i);
    exit(1);
    }
  }
 
 x = (float *) malloc(num * sizeof(float));
 if( !x)
  {
  printf("Cannot allocate x!\n");
  exit(1);
  }


 b = (float *) malloc(num * sizeof(float));
 if( !b)
  {
  printf("Cannot allocate b!\n");
  exit(1);
  }

 /* Now .. Filling the blanks */ 

 /* The initial values of Xs */
 for(i = 0; i < num; i++)
  fscanf(fp,"%f ", &x[i]);
 
 for(i = 0; i < num; i++)
 {
   for(j = 0; j < num; j++)
     fscanf(fp,"%f ",&a[i][j]);
   
   /* reading the b element */
   fscanf(fp,"%f ",&b[i]);
 }
 
 fclose(fp); 

}

  void print_floats(char *label, float *array, int size, int rank)
  {
    int i;
    printf("%s %d: ", label, rank);
    for(i = 0; i < size; i++)
    {
      printf("%f ", array[i]);
    }
    printf("\n");
  }

  void print_ints(char *label, int *array, int size, int rank)
  {
    int i;
    printf("%s %d: ", label, rank);
    for(i = 0; i < size; i++)
    {
      printf("%d ", array[i]);
    }
    printf("\n");
  }
/************************************************************/


void parallel_jacobi()
{
  int my_i, my_j, my_first_i, my_last_i; /* local indices */
  int rmndr; /* remainder if num is not divisible by comm_sz */
  int quotient; /* used to store num/comm_sz */
  float rel_err; /* compare relative error to err */
  int my_err; /* incremented if there is a relative error greater than err */
  int is_err; /* check to see if we can print and exit */
  int num_it = 0; /* number of iterations */
  float my_sum;
  float *my_x; //locally computed unknowns
  int my_n_count; /* local number of unknowns */
  int comm_sz; /* number of processes */
  int my_rank; /* my process rank */
  int *displ = NULL; /* for MPI_Allgatherv */
  int *recv_counts = NULL; /* for MPI_Allgatherv */
  int master_process = 0;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm new_comm; /* use in case there are more processes than unknowns */

  /* compute the local indices */

  rmndr = num % comm_sz;
  quotient = num/comm_sz;
  if(rmndr)
  {
    if(quotient)
    {
      if(my_rank < rmndr)
      {
        my_n_count = quotient + 1;
        my_first_i = my_rank * my_n_count;
      }
      else
      {
        my_n_count = quotient;
        my_first_i = my_rank * my_n_count + rmndr;
      }
      my_last_i = my_first_i + my_n_count;
      my_x = (float*)malloc(my_n_count * sizeof(float));
      recv_counts = (int*)malloc(comm_sz * sizeof(int)); 
      displ = (int*)malloc(comm_sz * sizeof(int)); 
      // MPI_Allgather((void*)&my_n_count, send_count, MPI_INT, (void*)recv_counts, send_count, MPI_INT, MPI_COMM_WORLD);
      int i;
      for(i = 0; i < comm_sz; i++)
      {
        if(i == 0)
        {
          displ[i] = 0;
          recv_counts[i] = quotient + 1;
        }
        else if(i < rmndr)
        {
          displ[i] = displ[i-1] + recv_counts[i-1];
          recv_counts[i] = quotient + 1;
        }
        else
        {
          displ[i] = displ[i-1] + recv_counts[i-1];
          recv_counts[i] = quotient;
        }
      }
    }
    else
    {
      MPI_Comm_split(MPI_COMM_WORLD, my_rank < num, my_rank, &new_comm);
      my_first_i = my_rank;
      my_last_i = (my_rank + 1);
      my_n_count = 1; //number of elements in subarray of unknowns
      my_x = (float*)malloc(my_n_count * sizeof(float));
    }
  }
  else
  {
    my_first_i = my_rank * quotient;
    my_last_i = (my_rank + 1) * quotient;
    my_n_count = quotient; //number of elements in subarray of unknowns
    my_x = (float*)malloc(my_n_count * sizeof(float));
  }

  /* jacobi algorith */

  while(TRUE)
  {
    my_err = 0;
    is_err = (my_rank != 0);
    if(my_rank < num)
    {
      for(my_i = my_first_i; my_i < my_last_i; my_i++)
      {
        my_sum = 0;
        for(my_j = 0; my_j < num; my_j++)
        {
          if(my_j != my_i)
          {
            my_sum += a[my_i][my_j] * x[my_j];
          }
        }
        my_x[my_i - my_first_i] = (b[my_i] - my_sum)/a[my_i][my_i];
        rel_err = fabs((my_x[my_i - my_first_i] - x[my_i])/my_x[my_i - my_first_i]);
        if(rel_err > err)
        {
          my_err++;
        }
      }
    }
    num_it++;
    int send_count = 1;
    MPI_Reduce((void*)&my_err, (void*)&is_err, send_count, MPI_INT, MPI_SUM, master_process, MPI_COMM_WORLD);
    if(!is_err)
    {
      if(my_rank == 0)
      {
        int i;
        for( i = 0; i < num; i++)
        {
          printf("%f\n",x[i]);
        }
        printf("total number of iterations: %d\n", num_it);
      }
      MPI_Abort(MPI_COMM_WORLD, -1);
      MPI_Finalize();
      return;
    }
    if(rmndr)
    {
      if(quotient)
      {
        MPI_Allgatherv((void*)my_x, my_n_count, MPI_FLOAT, (void*)x, recv_counts, displ, MPI_FLOAT, MPI_COMM_WORLD);
      }
      else
      {
        MPI_Allgather((void*)my_x, my_n_count, MPI_FLOAT, (void*)x, my_n_count, MPI_FLOAT, new_comm);
      }
    }
    else
    {
      MPI_Allgather((void*)my_x, my_n_count, MPI_FLOAT, (void*)x, my_n_count, MPI_FLOAT, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  printf("This should never happen!\n");
}



/************************************************************/


int main(int argc, char *argv[])
{
 // int nit = 0;  number of iterations 
 // int is_err = 0; /* set to 1 if there is a relative error greater than err */

 if( argc != 2)
 {
   printf("Usage: gsref filename\n");
   exit(1);
 }
  
 /* Read the input file and fill the global data structure above */ 
 get_input(argv[1]);
 
 /* Check for convergence condition */
 check_matrix();

 parallel_jacobi();

 return 0;

  // xnew = (float*)malloc(num * sizeof(float));
  // rerr = (float*)maloc(num * sizeof(float));
 
  // while(true)
  // {
  //   parallel_jacobi();
  //   nit++;

  //   for(i = 0; i < num; i++)
  //   {
  //     if(rerr[i] > err)
  //     {
  //       iserr = 1;
  //       free(x);
  //       x = xnew;
  //       break;
  //     }
  //   }
    
  //   if(iserr == 0)
  //   {
  //     /* Writing to the stdout */
  //     /* Keep that same format */
  //     for( i = 0; i < num; i++)
  //     {
  //       printf("%f\n",x[i]);
  //     }
       
  //     printf("total number of iterations: %d\n", nit);
       
  //     exit(0);
  //   }
  // }
}
