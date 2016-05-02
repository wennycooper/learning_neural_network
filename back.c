
#include "math.h"
#include "time.h"
#include "stdio.h"
#include "stdlib.h"

#define DIGITs 10
#define HIDDENs 15
#define PIXELs 20
#define H_PIXELs 4
#define V_PIXELs 5
#define TRAIN_FILE "train.pix"
#define TEST_FILE "test.pix"
#define f(x) (2/(1+exp(-x))-1)
#define df(f) ((1+f)*(1-f)/2)

static double v[PIXELs+1][HIDDENs+1];
static double w[HIDDENs+1][DIGITs+1];

void Init()
{
    unsigned i, j, k;

    for (j=0; j<=HIDDENs; j++)
        for (i=0; i<=DIGITs; i++)
            w[j][i] = 2.0 * (double) rand() / (double) RAND_MAX;

    for (k=0; k<=PIXELs; k++)
        for (j=0; j<=HIDDENs; j++)
            v[k][j] = 2.0 * (double) rand() / (double) RAND_MAX;

    return;
}

int Training()
{
    FILE *fp;
    char buffer[H_PIXELs+5];
    unsigned counter, item, i, j, k;
    double u, rate, threshold, err;
    double p[DIGITs+1], q[HIDDENs+1], t[DIGITs+1],
           x[PIXELs+1], z[HIDDENs+1], y[DIGITs+1];
    double dv[PIXELs+1][HIDDENs+1], dw[HIDDENs+1][DIGITs+1];
    counter = 0;

    if ((fp=fopen(TRAIN_FILE,"r")) == NULL) {
        printf("\n\t Can't open training file: %s\n",TRAIN_FILE);
        return 0;
    }
    printf("Input learning rate (0<alpha<1): "); scanf("%f", &rate);
    printf("Input Threshold: "); scanf("%f", &threshold);


/* Step 0 */
    Init();
    printf("\n Waiting for training... \n");

/* Step 1 */
    do {
        counter++;
        err = 0.0;
        fseek(fp, 0, SEEK_SET);
        for (k=0; k<=DIGITs; k++)
            for (j=0; j<=HIDDENs; j++)
                dw[j][k] = 0.0;
        for (j=0; j<=HIDDENs; j++)
            for (i=0; i<=PIXELs; i++)
                dv[i][j] = 0.0;

/* Step 2 */
        for (item=1; item<=DIGITs; item++) {
/* Step 3 */
            for (i=0; i<V_PIXELs; i++) {
                fgets(buffer, H_PIXELs+2, fp);
                for (j=1; j<=H_PIXELs; j++)
                    x[i*H_PIXELs+j] = (buffer[j-1] == '#') ?1 :-1;
            }
/* Step 4 */
            for (j=1; j<=HIDDENs; j++) {
                z[j]=v[0][j];
                for (i=1; i<=PIXELs; i++)
                    z[j] += (x[i] * v[i][j]);
                z[j] = f(z[j]);
            }
/* Step 5 */
            for (k=1; k<=DIGITs; k++) {
                y[k]=w[0][k];
                for (j=1; j<=HIDDENs; j++)
                    y[k] += (z[j] * w[j][k]);
                y[k] = f(y[k]);
            }

/* Step 6 */
            fgets(buffer, 3, fp);  /* get a target char */
            for (i=1; i<=DIGITs; i++) t[i] = -1.0;
            t[buffer[0]-'0'+1] = 1.0;
            for (k=1; k<=DIGITs; k++) {
                err += (t[k]-y[k]) * (t[k]-y[k]);
                p[k] = (t[k]-y[k]) * df(y[k]);
                for (j=0; j<=HIDDENs; j++)
                    dw[j][k] = rate * p[k] * z[j];
            }
/* Step 7 */
            for (j=1; j<=HIDDENs; j++) {
                q[j] = 0.0;
                for (k=1; k<=DIGITs; k++)
                    q[j] += (p[k] * w[j][k]);
                q[j] *= df(z[j]);
                for (i=0; i<=PIXELs; i++)
                    dv[i][j] = rate * q[j] * x[i];
            }
/* Step 8 */
            for (k=0; k<=DIGITs; k++)
                for (j=0; j<=HIDDENs; j++)
                    w[j][k] += dw[j][k];
            for (j=0; j<=HIDDENs; j++)
                for (i=0; i<=PIXELs; i++)
                    v[i][j] += dv[i][j];
        }
    } while ((err/2.0) > threshold);

    printf("\nAfter learning %u times, stable!\n",counter);
    fclose(fp);
    return 1;
}


int Testing()
{
    FILE *fp;
    char buffer[H_PIXELs+3];
    unsigned i, j, k;
    double t, x[PIXELs+1], z[HIDDENs+1], y[DIGITs+1];

    if ((fp=fopen(TEST_FILE,"r"))==NULL) {
        printf("\nCan't open test file: %s\n",TEST_FILE);
        return 0;
    }

    printf("\nThe result is ..\n");

    while (!feof(fp))  {
        fgets(buffer, 3, fp);
        for (i=0; i<V_PIXELs; i++) {
            fgets(buffer, PIXELs+2, fp);
            for (j=1; j<=H_PIXELs; j++)
                if (buffer[j-1] == '#' || buffer[j-1] == '@')
                    x[i*H_PIXELs+j] = 1;
                else if (buffer[j-1] == ' ')
                    x[i*H_PIXELs+j] = 0;
                else
                    x[i*H_PIXELs+j] = -1;
        }
        for (j=1; j<=HIDDENs; j++) {
            z[j] = v[0][j];
            for (i=1; i<=PIXELs; i++)
                z[j] += (x[i] * v[i][j]);
            z[j] = f(z[j]);
        }
        i=0;
        t=-1.0;
        for (k=1; k<=DIGITs; k++) {
            y[k] = w[0][k];
            for (j=1; j<=HIDDENs; j++)
                y[k] += (z[j] * w[j][k]);
            y[k] = f(y[k]);
            if (y[k] > 0 && y[k] > t) {
                t = y[k];
                i = k;
            }
        }
        printf(" %c.",(i==0) ?'x':(char) i+'0'-1);
    }
    fclose(fp);
}

int main()
{
    if (Training()) Testing();
}









