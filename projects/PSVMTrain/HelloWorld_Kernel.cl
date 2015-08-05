__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#define DIAGONAL 1
#define ONE_VERSUS_ALL 2
#define GET_INSTANCE 4

__kernel void multiply_matrices(
	__global int *mode, 
	__global int *attribute_count,
	__global int *instance_count,
	__global int *pivot, 
	__read_only image2d_t image, 
	__global float *buffer) {	

	int thread_index = get_global_id(0); 

	if(thread_index < *instance_count) {
		float sum = 0.f;
		int z, 
			width_attribute = (*attribute_count + (4 - (*attribute_count % 4))), 
			width_pixels = width_attribute/4,
			image_height = get_image_height(image),
			thread_column = (thread_index / image_height) * width_pixels,
			thread_row = thread_index % image_height;
		
		if(*mode == ONE_VERSUS_ALL) { //dot product of an instance with all others
			int
				pivot_column = (*pivot / image_height) * width_pixels,
				pivot_row = *pivot % image_height;
			
			for(z = 0; z < width_pixels; z++) {
				sum += dot(
					read_imagef(image, sampler, (int2)(pivot_column + z, pivot_row)), read_imagef(image, sampler, (int2)(thread_column + z, thread_row))
				);
			}
		} else if(*mode == DIAGONAL) { //dot product of an instance with itself, for all instances
			for(z = 0; z < width_pixels; z++) {
				float4 pixel = read_imagef(image, sampler, (int2)(thread_column + z, thread_row));
				sum += dot(pixel, pixel);
			}
		} /*else if(*mode == GET_INSTANCE) { //computes nothing; only returns instance pointed by pivot
			int 
				pixel_index = thread_column_index / 4, //position of pixel that contains required value
				channel_index = thread_column_index % 4; //channel is either red, green, blue or alpha

			float4 pixel = read_imagef(image, sampler, (int2)(pixel_index, *pivot));
			float values[4] = {pixel.x, pixel.y, pixel.z, pixel.w};
			sum = values[channel_index];
		}*/

		buffer[thread_index] = sum;
	} //else does nothing, nothing needed (in case it's outside dataset dimension)
}