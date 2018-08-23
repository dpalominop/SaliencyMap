#include "SaliencyMap.h"


int main(int argc, char const *argv[]){
    SaliencyMap map("images/oso.jpg");
    map.getData();
    double start_time = omp_get_wtime();
    map.run();
    double _time = omp_get_wtime() - start_time;
    map.showSalency();

    std::cout << "Tiempo de ejecucion: " << _time << std::endl; 

    return 0;
}
