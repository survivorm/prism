#include <sycl/sycl.hpp>
#include <cstdio>
int main(){
  sycl::device gpu; bool have=false;
  for(auto&d:sycl::device::get_devices()) if(d.is_gpu()){gpu=d;have=true;break;}
  if(!have) return 1;
  auto g=[&](auto k){ return gpu.get_info(k); };
  // EU count is not a standard SYCL info; use max_work_group_size + local mem as proxies
  size_t mwg = g(sycl::info::device::max_work_group_size());
  size_t lmem = g(sycl::info::device::local_mem_size());
  size_t gmem = g(sycl::info::device::global_mem_size());
  printf("max_work_group_size = %zu\n", mwg);
  printf("local_mem (per WG)  = %zu KB\n", lmem/1024);
  printf("global_mem          = %zu MB\n", gmem/1024/1024);
  printf("DEVICE_INFO_DONE\n");
  return 0;
}
