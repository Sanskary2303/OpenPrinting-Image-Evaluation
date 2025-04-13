#include <stdio.h>
#include <cups/cups.h>

int main() {
    printf("CUPS Test by Sanskar Yaduka\n");
    printf("CUPS Version: %s\n", cupsGetVersion());
    
    // List all available printers
    cups_dest_t *dests;
    int num_dests = cupsGetDests(&dests);
    printf("Found %d printers:\n", num_dests);
    
    for (int i = 0; i < num_dests; i++) {
        printf(" - %s\n", dests[i].name);
    }
    
    cupsFreeDests(num_dests, dests);
    return 0;
}
