#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <cuda.h>
#include "backproject.hcu"
#include <iostream>

using namespace std;

// from
// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__constant__ float geom_[MAX_PROJ_STACK_SIZE * 12];
__constant__ int3 proj_shape_;
__constant__ int3 vol_shape_;
__constant__ float3 vol_orig_;
__constant__ float3 voxel_size_;
static int3 proj_shape_host_;
static int3 vol_shape_host_;
texture<float, cudaTextureType2DLayered> projTex_;

inline __device__
float3 map( float3&& vp, int n )
{

   const float DSR = 700.0;
   const float DDR = 0;
   float theta = 0;
	

        if(n == 0){
                theta = -7.5;
        }else if(n == 1){
		theta = -6.43;
        }else if(n == 2){
		theta = -5.36;
        }else if(n == 3){
                theta = -4.29;
        }else if(n == 4){
                theta = -3.22;
        }else if(n == 5){
		theta = -2.15;
        }else if(n == 6){
		theta = -1.08;
        }else if(n == 7){
                theta = 0;
        }else if(n == 8){
		theta = 1.08;
        }else if(n == 9){
		theta = 2.15;
        }else if(n == 10){
		theta = 3.22;
        }else if(n == 11){
		theta = 4.29;
        }else if(n == 12){
		theta = 5.36;
        }else if(n == 13){
		theta = 6.43;
        }else if(n == 14){
		theta = 7.5;
        }

	// Transforma em radiano
	theta = (theta * M_PI)/180;

   // IMPLEMENTAÇÃO DO BRUNO!!!!
   const float* matrix = &(geom_[n*12]);
   float3 oi = make_float3(
	( vp.x + (( vp.z * ((DSR * std::sin(theta)) + vp.x)) / ((DSR * std::cos(theta)) + DDR - vp.z)) ),
	(vp.y * ((DSR * std::cos(theta))+DDR)) / ((DSR * std::cos(theta))+DDR- vp.z),
	((DSR * std::cos(theta))+DDR- vp.z)
   );

   
   ////////// IMPLEMENTAÇÃO DOS ALEMÃES!!!! ////////
/*   const float* matrix = &(geom_[n*12]);
   float3 oi = make_float3(
         (matrix[0] * vp.x + matrix[1] * vp.y + matrix[2] * vp.z + matrix[3]),// + tx,
         (matrix[4] * vp.x + matrix[5] * vp.y + matrix[6] * vp.z + matrix[7]),
         (matrix[8] * vp.x + matrix[9] * vp.y + matrix[10] * vp.z + matrix[11])

   );   
*/
   return oi;

}

__global__
void kernel_backproject( float* vol )
{
   const int i = blockIdx.x*blockDim.x + threadIdx.x;
   const int j = blockIdx.y*blockDim.y + threadIdx.y;
   const int k = blockIdx.z*blockDim.z + threadIdx.z;
   
   if( i >= vol_shape_.x  || j >= vol_shape_.y || k >= vol_shape_.z )
      return;

   const float x = i*voxel_size_.x + vol_orig_.x;
   const float y = j*voxel_size_.y + vol_orig_.y;
   const float z = k*voxel_size_.z + vol_orig_.z;

   float val = 0.0f;
   
   for( int n = 0; n < proj_shape_.z; ++n )
   {
      auto ip = map( make_float3( x, y, z ), n );

      ip.z = 1.0f / ip.z;
      //ip.x *= ip.z;
      //ip.y *= ip.z;

//      val += tex2DLayered( projTex_, (ip.x + 0.5 + 143.36)/0.14, (ip.y + 0.5)/0.14, n ) * ip.z * ip.z;	// bACKUP lINE
       val += tex2DLayered( projTex_, (ip.x + 0.5 + 35)/0.14, (ip.y + 0.5)/0.14, n ) * ip.z * ip.z;
   }

   // linear volume address
   const unsigned int l = vol_shape_.x * ( k*vol_shape_.y + j ) + i;
   vol[l] = val;
}

   
__global__
void kernel_project( const float* vol, float* proj )
{
   const int i = blockIdx.x*blockDim.x + threadIdx.x;
   const int j = blockIdx.y*blockDim.y + threadIdx.y;
   const int k = blockIdx.z*blockDim.z + threadIdx.z;

   if( i >= vol_shape_.x  || j >= vol_shape_.y || k >= vol_shape_.z )
      return;

   const float x = i*voxel_size_.x + vol_orig_.x;
   const float y = j*voxel_size_.y + vol_orig_.y;
   const float z = k*voxel_size_.z + vol_orig_.z;

   const float v = vol[vol_shape_.x * ( k*vol_shape_.y + j ) + i];

   for( int n = 0; n < proj_shape_.z; ++n )
   {
      float3 ip = map( make_float3( x, y, z ), n );

      //ip.x *= 1.0f / ip.z;
      //ip.y *= 1.0f / ip.z;

      const float vz = v / (ip.z*ip.z);

      // four neighbours on projection
      const int u1 = ((int)ip.x),
                v1 = ((int)ip.y);
      const int u2 = u1+1,
                v2 = v1+1;

      // simulate cudaAddressModeBorder
      if( u1 >= -1 && v1 >= -1  && u2 <= proj_shape_.x && v2 <= proj_shape_.y )
      {
         const float wu2 = ip.x - ((float)u1);
         const float wu1 = 1.0f - wu2;
         const float wv2 = ip.y - ((float)v1);
         const float wv1 = 1.0f - wv2;

         const unsigned int l1 = proj_shape_.x * ( n*proj_shape_.y + v1 ) + u1;
         const unsigned int l2 = l1 + proj_shape_.x;

         if( u1 >= 0 )
         {
            const float vzwu1 = vz*wu1;
            if( v1 >= 0 )
               atomicAdd( &proj[l1], vzwu1*wv1 );
            if( v2 < proj_shape_.y )
               atomicAdd( &proj[l2], vzwu1*wv2 );
         }
         if( u2 < proj_shape_.x )
         {
            const float vzwu2 = vz*wu2;
            if( v1 >= 0 )
               atomicAdd( &proj[l1 + 1], vzwu2*wv1 );
            if( v2 < proj_shape_.y )
               atomicAdd( &proj[l2 + 1], vzwu2*wv2 );
         }

      }
   }
}

__host__
void cuda_init_backproject( float* geom,
                            int U, int V, int N,
                            int X, int Y, int Z,
                            float ox, float oy, float oz,
                            float sx, float sy, float sz )
{
   proj_shape_host_ = make_int3( U, V, N );
   vol_shape_host_ = make_int3( X, Y, Z );
   auto vol_orig = make_float3( ox, oy, oz );
   auto voxel_size = make_float3( sx, sy, sz );

   gpuErrchk( cudaMemcpyToSymbol( geom_, geom, 12 * sizeof(float) * N ) );
   gpuErrchk( cudaMemcpyToSymbol( proj_shape_, &proj_shape_host_, sizeof(int3) ) );
   gpuErrchk( cudaMemcpyToSymbol( vol_shape_, &vol_shape_host_, sizeof(int3) ) );
   gpuErrchk( cudaMemcpyToSymbol( vol_orig_, &vol_orig, sizeof(float3) ) );
   gpuErrchk( cudaMemcpyToSymbol( voxel_size_, &voxel_size, sizeof(float3) ) );
}

__host__
void cuda_backproject( const float* proj, float* vol )
{
   // set texture properties
   projTex_.addressMode[0] = cudaAddressModeBorder;
   projTex_.addressMode[1] = cudaAddressModeBorder;
   projTex_.addressMode[2] = cudaAddressModeBorder;
   projTex_.filterMode = cudaFilterModeLinear;
   projTex_.normalized = false;

   // malloc cuda array for texture
   cudaExtent projExtent = make_cudaExtent( proj_shape_host_.x,
                                            proj_shape_host_.y,
                                            proj_shape_host_.z );
   cudaArray *projArray;
   static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
   gpuErrchk( cudaMalloc3DArray( &projArray, &channelDesc, projExtent, cudaArrayLayered ) );


   // copy data to 3D array
   cudaMemcpy3DParms copyParams = {0};
   copyParams.srcPtr   = make_cudaPitchedPtr( const_cast<float*>( proj ),
                                              proj_shape_host_.x*sizeof(float),
                                              proj_shape_host_.x,
                                              proj_shape_host_.y
                                            );
   copyParams.dstArray = projArray;
   copyParams.extent   = projExtent;
   copyParams.kind     = cudaMemcpyDeviceToDevice;
   gpuErrchk( cudaMemcpy3D( &copyParams ) );
   
   // bind texture reference
   gpuErrchk( cudaBindTextureToArray( projTex_, (cudaArray*)projArray,
            channelDesc ) );

   // launch kernel
   const unsigned int gridsize_x = (vol_shape_host_.x-1) / BLOCKSIZE_X + 1;
   const unsigned int gridsize_y = (vol_shape_host_.y-1) / BLOCKSIZE_Y + 1;
   const unsigned int gridsize_z = (vol_shape_host_.z-1) / BLOCKSIZE_Z + 1;
   const dim3 grid = dim3( gridsize_x, gridsize_y, gridsize_z );
   const dim3 block = dim3( BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z );
   fprintf(stderr, "\n\n\n\n");
   fprintf(stderr, "Iniciano o backproject");
   fprintf(stderr, "Saiu do backproject");
   fprintf(stderr, "\n\n\n\n");
   kernel_backproject<<< grid, block >>>( vol );

   // check for errors
   gpuErrchk( cudaPeekAtLastError() );
   gpuErrchk( cudaDeviceSynchronize() );

   // cleanup
   gpuErrchk( cudaUnbindTexture( projTex_ ) );
   gpuErrchk( cudaFreeArray( projArray ) );
}

   
__host__
void cuda_project( const float* vol, float* proj )
{
   // set proj to zero
   cudaMemset( proj, 0, proj_shape_host_.x*proj_shape_host_.y*proj_shape_host_.z
         * sizeof( float ) );

   // launch kernel
   const unsigned int gridsize_x = (vol_shape_host_.x-1) / BLOCKSIZE_X + 1;
   const unsigned int gridsize_y = (vol_shape_host_.y-1) / BLOCKSIZE_Y + 1;
   const unsigned int gridsize_z = (vol_shape_host_.z-1) / BLOCKSIZE_Z + 1;
   const dim3 grid = dim3( gridsize_x, gridsize_y, gridsize_z );
   const dim3 block = dim3( BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z );
   kernel_project<<< grid, block >>>( vol, proj );
}

#endif


