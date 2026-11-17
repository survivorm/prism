#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>
#include <unistd.h>
int main(){
  setvbuf(stdout,NULL,_IOLBF,0);
  sycl::device gpu; bool have=false;
  for(auto&d:sycl::device::get_devices()) if(d.is_gpu()){gpu=d;have=true;break;}
  if(!have) return 1;
  sycl::queue q{gpu, sycl::property::queue::in_order()};
  int *R=sycl::malloc_device<int>(2,q);
  q.submit([&](sycl::handler&h){
    sycl::local_accessor<float,1> la0(sycl::range<1>(4),h);
    sycl::local_accessor<float,1> la1(sycl::range<1>(4),h);
    h.parallel_for(sycl::nd_range<2>(sycl::range<2>(1,4),sycl::range<2>(1,4)),
      [=](sycl::nd_item<2> it){
        size_t lid=it.get_global_id(1);
        la0[lid]=100.0f+float(lid);
        la1[lid]=200.0f+float(lid);
        sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
        if (lid==0){ R[0]=int(la0[1]); R[1]=int(la1[1]); }
      });
  }).wait_and_throw();
  std::vector<int> r(2);
  q.submit([&](sycl::handler&h){h.copy(R,r.data(),2);}).wait_and_throw();
  std::printf("la0[1]=%d (expect 101)  la1[1]=%d (expect 201)\n", r[0], r[1]);
  // Теперь проверка КОПИИ: la1 = la0 (copy) — общая ли память?
  int *R2=sycl::malloc_device<int>(1,q);
  q.submit([&](sycl::handler&h){
    sycl::local_accessor<float,1> la0(sycl::range<1>(4),h);
    auto la1 = la0;  // copy
    h.parallel_for(sycl::nd_range<2>(sycl::range<2>(1,4),sycl::range<2>(1,4)),
      [=](sycl::nd_item<2> it){
        size_t lid=it.get_global_id(1);
        la0[lid]=42.0f+float(lid);
        sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
        if (lid==0) R2[0]=int(la1[1]); // 43 (общая память) или 0 (разная)?
      });
  }).wait_and_throw();
  std::vector<int> r2(1);
  q.submit([&](sycl::handler&h){h.copy(R2,r2.data(),1);}).wait_and_throw();
  std::printf("copy: la1[1]=%d (43 = общая память, 0 = РАЗНЫЕ выделение)\n", r2[0]);
  std::printf("COPY_TEST_DONE\n");
  return 0;
}
