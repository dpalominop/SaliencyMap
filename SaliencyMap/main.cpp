#include "SaliencyMap.h"


int main(int argc, char const *argv[]){
    salencyMap map("images/oso.jpg");
    map.getData();
    map.run();
    return 0;
}
