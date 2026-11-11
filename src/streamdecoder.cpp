// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <string.h>

static int isFilterValid(meshopt_BufferViewFilter filter, size_t stride)
{
	switch (filter)
	{
	case meshopt_BufferViewFilter_None:
		return stride > 0;

	case meshopt_BufferViewFilter_Octahedral:
		return stride == 4 || stride == 8;

	case meshopt_BufferViewFilter_Quaternion:
		return stride == 8;

	case meshopt_BufferViewFilter_Exponential:
		return stride > 0 && stride % 4 == 0;

	case meshopt_BufferViewFilter_Color:
		return stride == 4 || stride == 8;

	default:
		return 0;
	}
}

static void decodeFilter(void* destination, size_t count, size_t stride, meshopt_BufferViewFilter filter)
{
	switch (filter)
	{
	case meshopt_BufferViewFilter_Octahedral:
		meshopt_decodeFilterOct(destination, count, stride);
		break;

	case meshopt_BufferViewFilter_Quaternion:
		meshopt_decodeFilterQuat(destination, count, stride);
		break;

	case meshopt_BufferViewFilter_Exponential:
		meshopt_decodeFilterExp(destination, count, stride);
		break;

	case meshopt_BufferViewFilter_Color:
		meshopt_decodeFilterColor(destination, count, stride);
		break;

	default:
		break;
	}
}

int meshopt_decodeBufferViewChunk(void* destination, const struct meshopt_BufferView* view, size_t element_offset, size_t element_count)
{
	if (!view || !view->stride)
		return -4;

	if (element_offset > view->count || element_count > view->count - element_offset)
		return -4;

	if (!destination && element_count)
		return -4;

	if (view->count > size_t(-1) / view->stride)
		return -4;

	size_t decoded_size = view->count * view->stride;
	int has_compressed = view->data && view->size;

	if (!has_compressed)
	{
		if (decoded_size && (!view->fallback || view->fallback_size < decoded_size))
			return -4;

		if (element_count)
		{
			const unsigned char* fallback = static_cast<const unsigned char*>(view->fallback) + element_offset * view->stride;
			memcpy(destination, fallback, element_count * view->stride);
		}

		return 0;
	}

	switch (view->mode)
	{
	case meshopt_BufferViewMode_Attributes:
	{
		if (view->stride > 256 || view->stride % 4 != 0 || !isFilterValid(view->filter, view->stride))
			return -4;

		int result = meshopt_decodeVertexBufferChunk(destination, element_offset, element_count, view->count, view->stride, view->data, view->size);
		if (result != 0)
			return result;

		if (element_count)
			decodeFilter(destination, element_count, view->stride, view->filter);

		return 0;
	}

	case meshopt_BufferViewMode_Triangles:
		if (view->filter != meshopt_BufferViewFilter_None || view->count % 3 != 0 || (view->stride != 2 && view->stride != 4))
			return -4;

		return meshopt_decodeIndexBufferChunk(destination, element_offset, element_count, view->count, view->stride, view->data, view->size);

	case meshopt_BufferViewMode_Indices:
		if (view->filter != meshopt_BufferViewFilter_None || (view->stride != 2 && view->stride != 4))
			return -4;

		return meshopt_decodeIndexSequenceChunk(destination, element_offset, element_count, view->count, view->stride, view->data, view->size);

	default:
		return -4;
	}
}
