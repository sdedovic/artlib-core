#define PI_F  3.141592654f
#define TAU_F 6.283185307f

extern "C"
/**
 * For each Point stored in the supplied buffer apply brownian motion by moving it
 * @param pointBuffer
 * @param randomNumberBuffer
 * @param countPoints
 * @param magnitude
 */
__global__ void moveBrownian(float *pointBuffer, float *randomNumberBuffer, int countPoints, float magnitude)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= countPoints)
        return;

    float theta = randomNumberBuffer[index] * TAU_F;
    pointBuffer[index] += magnitude * cos(theta);
    pointBuffer[index + countPoints] += magnitude * sin(theta);
}

extern "C"
/**
 * For each Point stored in the pointBuffer apply an inverse squared
 * @param pointBuffer
 * @param countPoints
 * @param forceConst
 */
__global__ void applyForceField(float *pointBuffer, int countPoints, float forceConst)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= countPoints)
        return;

    float xAmt = 0.0f;
    float yAmt = 0.0f;
    for(int i = 0; i < countPoints; i++) {
        if (i != index) {
            float diffX = pointBuffer[i] - pointBuffer[index];
            float diffY = pointBuffer[i + countPoints] - pointBuffer[index + countPoints];
            float theta = atan2(diffY, diffX);

            float dist = sqrtf(powf(diffX, 2) + powf(diffY, 2));
            float mag = forceConst / powf(dist, 2);

            xAmt += mag * cos(theta);
            yAmt += mag * sin(theta);
        }
    }

    pointBuffer[index] += xAmt;
    pointBuffer[index + countPoints] += yAmt;
}

extern "C"
/**
 * For each Point stored in the pointBuffer move it by the amount in the moveByBuffer buffer.
 *  X and Y coordinates are moved independently.
 * @param pointBuffer
 * @param moveByBuffer
 * @param countPoints
 */
__global__ void moveEach(float *pointBuffer, float *moveByBuffer, int countPoints)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= countPoints)
        return;

    int yIndex = index + countPoints;
    pointBuffer[index] += moveByBuffer[index];
    pointBuffer[yIndex] += moveByBuffer[yIndex];
}

extern "C"
/**
 * For all Points stored in the pointBuffer move it by the supplied distances.
 * @param pointBuffer
 * @param xDistance
 * @param yDistance
 * @param countPoints
 */
__global__ void moveAll(float *pointBuffer, float xDistance, float yDistance, int countPoints)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= countPoints)
        return;

    pointBuffer[index] += xDistance;
    pointBuffer[index + countPoints] += yDistance;
}