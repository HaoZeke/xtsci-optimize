#include <stdio.h>
#include "xts.h"

int main(void) {
    printf("%s\n", xts_version());
    xts_abi_stamp_t stamp = xts_abi_stamp();
    return xts_abi_compatible(&stamp) == 0;
}
