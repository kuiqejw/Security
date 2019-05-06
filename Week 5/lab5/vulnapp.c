/*
 * Vulnapp. Vulnerable app for buffer overflow testing.
 *
 * usage: 
 *      echo hello | ./vulnapp
 *      ./vulnapp < hello.txt
 * 
 * To compile:
 * 	Run ./make_vulnapp
 *
 * Revised by Nils, SUTD, 2018, Z. TANG, SUTD, 2019.
 * Based on the original:
 * Copyright 2004 by Feckless C. Coder, PhD.
 */

#include <stdio.h>
#include <string.h>
#define INPUT_BUFFER 64 /* maximum name size */

/*
 * read input, copy into s
 */
void getlines(char *s)
{
  int c;
  while (1){
    c=getchar();
    // terminate input at linebreak or EOF. This allows direct input by user, or single-line pattern
    if(c=='\n'||c==EOF)
      break;
    *s++ = c;
  }
  *s = '\0';
}

int main()
{
  char input[INPUT_BUFFER];
  printf("Please put in the text, terminated with \\n character:\n");
  
  getlines(input);

  // the following will stop printing at string termination character \x00
  printf("Your text is: %s\n",input);
  return 0;
}


