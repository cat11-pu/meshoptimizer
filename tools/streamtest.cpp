#include "../src/meshoptimizer.h"

#define CGLTF_IMPLEMENTATION
#include "../extern/cgltf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

static unsigned int gSeed = 0x12345678;

static unsigned int nextRandom()
{
	gSeed ^= gSeed << 13;
	gSeed ^= gSeed >> 17;
	gSeed ^= gSeed << 5;
	return gSeed;
}

static size_t minSize(size_t a, size_t b)
{
	return a < b ? a : b;
}

static int compareBytes(const char* name, const std::vector<unsigned char>& actual, const std::vector<unsigned char>& expected)
{
	if (actual.size() != expected.size())
	{
		printf("FAIL %s: size mismatch %zu vs %zu\n", name, actual.size(), expected.size());
		return 1;
	}

	if (!actual.empty() && memcmp(actual.data(), expected.data(), actual.size()) != 0)
	{
		size_t first = 0;
		while (first < actual.size() && actual[first] == expected[first])
			++first;

		printf("FAIL %s: byte mismatch at %zu (%02x vs %02x)\n", name, first, actual[first], expected[first]);
		return 1;
	}

	return 0;
}

static unsigned int readIndex(const unsigned char* data, size_t index_size, size_t index)
{
	if (index_size == 2)
		return reinterpret_cast<const unsigned short*>(data)[index];
	else
		return reinterpret_cast<const unsigned int*>(data)[index];
}

static int compareFallback(const char* name, const std::vector<unsigned char>& actual, const std::vector<unsigned char>& expected, meshopt_BufferViewMode mode, size_t index_size)
{
	if (mode != meshopt_BufferViewMode_Triangles)
		return compareBytes(name, actual, expected);

	size_t triangle_count = actual.size() / index_size / 3;

	for (size_t i = 0; i < triangle_count; ++i)
	{
		unsigned int a[3], b[3];

		for (size_t k = 0; k < 3; ++k)
		{
			a[k] = readIndex(actual.data(), index_size, i * 3 + k);
			b[k] = readIndex(expected.data(), index_size, i * 3 + k);
		}

		int shift_a = a[0] <= a[1] && a[0] <= a[2] ? 0 : (a[1] <= a[2] ? 1 : 2);
		int shift_b = b[0] <= b[1] && b[0] <= b[2] ? 0 : (b[1] <= b[2] ? 1 : 2);

		for (size_t k = 0; k < 3; ++k)
		{
			if (a[(k + shift_a) % 3] != b[(k + shift_b) % 3])
			{
				printf("FAIL %s: triangle %zu mismatch (%u %u %u vs %u %u %u)\n", name, i,
				    a[0], a[1], a[2], b[0], b[1], b[2]);
				return 1;
			}
		}
	}

	return 0;
}

static int testVertexChunks(size_t vertex_count, size_t vertex_size)
{
	std::vector<unsigned char> vertices(vertex_count * vertex_size);
	for (size_t i = 0; i < vertices.size(); ++i)
		vertices[i] = (unsigned char)(nextRandom() & 0xff);

	std::vector<unsigned char> encoded(meshopt_encodeVertexBufferBound(vertex_count, vertex_size));
	encoded.resize(meshopt_encodeVertexBuffer(&encoded[0], encoded.size(), &vertices[0], vertex_count, vertex_size));
	if (encoded.empty())
	{
		printf("FAIL vertex encode %zu/%zu\n", vertex_count, vertex_size);
		return 1;
	}

	std::vector<unsigned char> reference(vertex_count * vertex_size);
	int result = meshopt_decodeVertexBuffer(&reference[0], vertex_count, vertex_size, &encoded[0], encoded.size());
	if (result != 0)
	{
		printf("FAIL vertex decode %zu/%zu: %d\n", vertex_count, vertex_size, result);
		return 1;
	}

	size_t failures = 0;
	size_t offsets[] = {0, minSize(1, vertex_count), minSize(7, vertex_count), minSize(15, vertex_count), minSize(16, vertex_count), minSize(17, vertex_count), vertex_count / 2, vertex_count ? vertex_count - 1 : 0};

	for (size_t oi = 0; oi < sizeof(offsets) / sizeof(offsets[0]); ++oi)
	{
		size_t offset = offsets[oi] > vertex_count ? vertex_count : offsets[oi];
		size_t chunk_sizes[] = {minSize(1, vertex_count - offset), minSize(13, vertex_count - offset), minSize(64, vertex_count - offset), minSize(256, vertex_count - offset), vertex_count - offset};

		for (size_t ci = 0; ci < sizeof(chunk_sizes) / sizeof(chunk_sizes[0]); ++ci)
		{
			size_t count = chunk_sizes[ci];
			if (!count && offset)
				continue;

			std::vector<unsigned char> chunk(count * vertex_size);
			int rc = meshopt_decodeVertexBufferChunk(&chunk[0], offset, count, vertex_count, vertex_size, &encoded[0], encoded.size());
			if (rc != 0)
			{
				printf("FAIL vertex chunk %zu/%zu offset %zu count %zu: %d\n", vertex_count, vertex_size, offset, count, rc);
				++failures;
				continue;
			}

			std::vector<unsigned char> expected(reference.begin() + offset * vertex_size, reference.begin() + (offset + count) * vertex_size);
			failures += compareBytes("vertex chunk", chunk, expected);
		}
	}

	return int(failures);
}

static int testIndexChunks(size_t index_count, size_t index_size)
{
	std::vector<unsigned int> indices(index_count);
	for (size_t i = 0; i < index_count; ++i)
		indices[i] = nextRandom() % 4096;

	std::vector<unsigned char> encoded(meshopt_encodeIndexBufferBound(index_count, 4096));
	encoded.resize(meshopt_encodeIndexBuffer(&encoded[0], encoded.size(), &indices[0], index_count));
	if (encoded.empty())
	{
		printf("FAIL index encode %zu/%zu\n", index_count, index_size);
		return 1;
	}

	size_t output_count = index_count;
	std::vector<unsigned char> reference(output_count * index_size);
	int result = meshopt_decodeIndexBuffer(&reference[0], index_count, index_size, &encoded[0], encoded.size());
	if (result != 0)
	{
		printf("FAIL index decode %zu/%zu: %d\n", index_count, index_size, result);
		return 1;
	}

	size_t failures = 0;
	size_t offsets[] = {0, minSize(3, index_count), minSize(6, index_count), minSize(15, index_count), index_count / 6 * 3, index_count ? index_count - 3 : 0};

	for (size_t oi = 0; oi < sizeof(offsets) / sizeof(offsets[0]); ++oi)
	{
		size_t offset = offsets[oi] > index_count ? index_count : offsets[oi];
		offset = offset / 3 * 3;
		size_t chunk_sizes[] = {minSize(3, index_count - offset), minSize(12, index_count - offset), minSize(96, index_count - offset), index_count - offset};

		for (size_t ci = 0; ci < sizeof(chunk_sizes) / sizeof(chunk_sizes[0]); ++ci)
		{
			size_t count = chunk_sizes[ci] / 3 * 3;
			if (!count && offset)
				continue;

			std::vector<unsigned char> chunk(count * index_size);
			int rc = meshopt_decodeIndexBufferChunk(&chunk[0], offset, count, index_count, index_size, &encoded[0], encoded.size());
			if (rc != 0)
			{
				printf("FAIL index chunk %zu/%zu offset %zu count %zu: %d\n", index_count, index_size, offset, count, rc);
				++failures;
				continue;
			}

			std::vector<unsigned char> expected(reference.begin() + offset * index_size, reference.begin() + (offset + count) * index_size);
			failures += compareBytes("index chunk", chunk, expected);
		}
	}

	return int(failures);
}

static int testSequenceChunks(size_t index_count, size_t index_size)
{
	std::vector<unsigned int> indices(index_count);
	for (size_t i = 0; i < index_count; ++i)
		indices[i] = nextRandom() % 4096;

	std::vector<unsigned char> encoded(meshopt_encodeIndexSequenceBound(index_count, 4096));
	encoded.resize(meshopt_encodeIndexSequence(&encoded[0], encoded.size(), &indices[0], index_count));
	if (encoded.empty())
	{
		printf("FAIL sequence encode %zu/%zu\n", index_count, index_size);
		return 1;
	}

	std::vector<unsigned char> reference(index_count * index_size);
	int result = meshopt_decodeIndexSequence(&reference[0], index_count, index_size, &encoded[0], encoded.size());
	if (result != 0)
	{
		printf("FAIL sequence decode %zu/%zu: %d\n", index_count, index_size, result);
		return 1;
	}

	size_t failures = 0;
	size_t offsets[] = {0, minSize(1, index_count), minSize(7, index_count), index_count / 2, index_count ? index_count - 1 : 0};

	for (size_t oi = 0; oi < sizeof(offsets) / sizeof(offsets[0]); ++oi)
	{
		size_t offset = offsets[oi] > index_count ? index_count : offsets[oi];
		size_t chunk_sizes[] = {minSize(1, index_count - offset), minSize(17, index_count - offset), minSize(128, index_count - offset), index_count - offset};

		for (size_t ci = 0; ci < sizeof(chunk_sizes) / sizeof(chunk_sizes[0]); ++ci)
		{
			size_t count = chunk_sizes[ci];
			if (!count && offset)
				continue;

			std::vector<unsigned char> chunk(count * index_size);
			int rc = meshopt_decodeIndexSequenceChunk(&chunk[0], offset, count, index_count, index_size, &encoded[0], encoded.size());
			if (rc != 0)
			{
				printf("FAIL sequence chunk %zu/%zu offset %zu count %zu: %d\n", index_count, index_size, offset, count, rc);
				++failures;
				continue;
			}

			std::vector<unsigned char> expected(reference.begin() + offset * index_size, reference.begin() + (offset + count) * index_size);
			failures += compareBytes("sequence chunk", chunk, expected);
		}
	}

	return int(failures);
}

static int testFilterChunk(meshopt_BufferViewFilter filter, size_t stride, size_t count, int bits)
{
	std::vector<float> source(count * 4);
	for (size_t i = 0; i < source.size(); ++i)
		source[i] = float(int(nextRandom() % 65535) - 32768) / 32768.f;

	std::vector<unsigned char> filtered(count * stride);
	switch (filter)
	{
	case meshopt_BufferViewFilter_Octahedral:
		meshopt_encodeFilterOct(&filtered[0], count, stride, bits, &source[0]);
		break;
	case meshopt_BufferViewFilter_Quaternion:
		meshopt_encodeFilterQuat(&filtered[0], count, stride, bits, &source[0]);
		break;
	case meshopt_BufferViewFilter_Exponential:
		meshopt_encodeFilterExp(&filtered[0], count, stride, bits, &source[0], meshopt_EncodeExpSeparate);
		break;
	case meshopt_BufferViewFilter_Color:
		meshopt_encodeFilterColor(&filtered[0], count, stride, bits, &source[0]);
		break;
	default:
		return 0;
	}

	std::vector<unsigned char> encoded(meshopt_encodeVertexBufferBound(count, stride));
	encoded.resize(meshopt_encodeVertexBuffer(&encoded[0], encoded.size(), &filtered[0], count, stride));
	if (encoded.empty())
	{
		printf("FAIL filter encode\n");
		return 1;
	}

	meshopt_BufferView view = {};
	view.data = &encoded[0];
	view.size = encoded.size();
	view.count = count;
	view.stride = stride;
	view.mode = meshopt_BufferViewMode_Attributes;
	view.filter = filter;

	std::vector<unsigned char> reference(count * stride);
	int rc = meshopt_decodeBufferViewChunk(&reference[0], &view, 0, count);
	if (rc != 0)
	{
		printf("FAIL filter decode: %d\n", rc);
		return 1;
	}

	size_t failures = 0;
	for (size_t offset = 0; offset < count; offset += 37)
	{
		size_t chunk_count = minSize(37, count - offset);
		std::vector<unsigned char> chunk(chunk_count * stride);

		rc = meshopt_decodeBufferViewChunk(&chunk[0], &view, offset, chunk_count);
		if (rc != 0)
		{
			printf("FAIL filter chunk offset %zu: %d\n", offset, rc);
			++failures;
			continue;
		}

		std::vector<unsigned char> expected(reference.begin() + offset * stride, reference.begin() + (offset + chunk_count) * stride);
		failures += compareBytes("filter chunk", chunk, expected);
	}

	return int(failures);
}

static int testCodecs()
{
	int failures = 0;
	size_t strides[] = {4, 8, 12, 16, 32, 64, 256};
	size_t counts[] = {1, 15, 16, 17, 255, 256, 257, 513, 1000};

	for (size_t si = 0; si < sizeof(strides) / sizeof(strides[0]); ++si)
		for (size_t ci = 0; ci < sizeof(counts) / sizeof(counts[0]); ++ci)
			failures += testVertexChunks(counts[ci], strides[si]);

	size_t triangle_counts[] = {3, 6, 48, 96, 3000};
	for (size_t ci = 0; ci < sizeof(triangle_counts) / sizeof(triangle_counts[0]); ++ci)
	{
		failures += testIndexChunks(triangle_counts[ci], 2);
		failures += testIndexChunks(triangle_counts[ci], 4);
	}

	size_t sequence_counts[] = {1, 17, 300, 1000};
	for (size_t ci = 0; ci < sizeof(sequence_counts) / sizeof(sequence_counts[0]); ++ci)
	{
		failures += testSequenceChunks(sequence_counts[ci], 2);
		failures += testSequenceChunks(sequence_counts[ci], 4);
	}

	failures += testFilterChunk(meshopt_BufferViewFilter_Octahedral, 4, 311, 8);
	failures += testFilterChunk(meshopt_BufferViewFilter_Octahedral, 8, 311, 16);
	failures += testFilterChunk(meshopt_BufferViewFilter_Quaternion, 8, 311, 16);
	failures += testFilterChunk(meshopt_BufferViewFilter_Exponential, 16, 311, 16);
	failures += testFilterChunk(meshopt_BufferViewFilter_Color, 4, 311, 8);
	failures += testFilterChunk(meshopt_BufferViewFilter_Color, 8, 311, 16);

	printf("%s codec chunk tests (%d failures)\n", failures ? "FAIL" : "PASS", failures);
	return failures;
}

static int testGltfFile(const char* path)
{
	cgltf_options options = {};
	cgltf_data* data = NULL;

	cgltf_result result = cgltf_parse_file(&options, path, &data);
	if (result != cgltf_result_success)
	{
		printf("FAIL %s: parse %d\n", path, int(result));
		return 1;
	}

	result = cgltf_load_buffers(&options, data, path);
	if (result != cgltf_result_success)
	{
		printf("FAIL %s: load buffers %d\n", path, int(result));
		cgltf_free(data);
		return 1;
	}

	result = cgltf_validate(data);
	if (result != cgltf_result_success)
	{
		printf("FAIL %s: validate %d\n", path, int(result));
		cgltf_free(data);
		return 1;
	}

	int failures = 0;
	size_t compressed_views = 0;
	size_t fallback_views = 0;

	for (size_t i = 0; i < data->buffer_views_count; ++i)
	{
		cgltf_buffer_view* bv = &data->buffer_views[i];
		if (!bv->has_meshopt_compression)
			continue;

		cgltf_meshopt_compression* mc = &bv->meshopt_compression;
		if (!mc->buffer->data)
		{
			printf("FAIL %s: view %zu has no compressed data\n", path, i);
			++failures;
			continue;
		}

		++compressed_views;

		const unsigned char* compressed = static_cast<const unsigned char*>(mc->buffer->data) + mc->offset;
		const unsigned char* fallback = bv->buffer && bv->buffer->data ? static_cast<const unsigned char*>(bv->buffer->data) + bv->offset : NULL;
		size_t fallback_size = fallback ? bv->size : 0;
		if (fallback)
			++fallback_views;

		meshopt_BufferView view = {};
		view.data = compressed;
		view.size = mc->size;
		view.fallback = fallback;
		view.fallback_size = fallback_size;
		view.count = mc->count;
		view.stride = mc->stride;
		view.mode = meshopt_BufferViewMode(mc->mode);
		view.filter = meshopt_BufferViewFilter(mc->filter);

		size_t decoded_size = mc->count * mc->stride;
		std::vector<unsigned char> full(decoded_size);

		switch (mc->mode)
		{
		case cgltf_meshopt_compression_mode_attributes:
			result = cgltf_result(meshopt_decodeVertexBuffer(&full[0], mc->count, mc->stride, compressed, mc->size));
			break;
		case cgltf_meshopt_compression_mode_triangles:
			result = cgltf_result(meshopt_decodeIndexBuffer(&full[0], mc->count, mc->stride, compressed, mc->size));
			break;
		case cgltf_meshopt_compression_mode_indices:
			result = cgltf_result(meshopt_decodeIndexSequence(&full[0], mc->count, mc->stride, compressed, mc->size));
			break;
		default:
			result = cgltf_result_invalid_gltf;
			break;
		}

		if (result != cgltf_result_success)
		{
			printf("FAIL %s: view %zu full decode %d\n", path, i, int(result));
			++failures;
			continue;
		}

		switch (mc->filter)
		{
		case cgltf_meshopt_compression_filter_octahedral:
			meshopt_decodeFilterOct(&full[0], mc->count, mc->stride);
			break;
		case cgltf_meshopt_compression_filter_quaternion:
			meshopt_decodeFilterQuat(&full[0], mc->count, mc->stride);
			break;
		case cgltf_meshopt_compression_filter_exponential:
			meshopt_decodeFilterExp(&full[0], mc->count, mc->stride);
			break;
		case cgltf_meshopt_compression_filter_color:
			meshopt_decodeFilterColor(&full[0], mc->count, mc->stride);
			break;
		default:
			break;
		}

		std::vector<unsigned char> whole(decoded_size);
		int rc = meshopt_decodeBufferViewChunk(&whole[0], &view, 0, mc->count);
		if (rc != 0)
		{
			printf("FAIL %s: view %zu chunk full decode %d\n", path, i, rc);
			++failures;
			continue;
		}

		if (compareBytes("bufferView full", whole, full))
			++failures;

		std::vector<unsigned char> assembled(decoded_size);
		size_t target_chunk = 4096 / mc->stride;
		if (mc->mode == cgltf_meshopt_compression_mode_triangles)
			target_chunk = target_chunk / 3 * 3;
		if (target_chunk == 0)
			target_chunk = mc->mode == cgltf_meshopt_compression_mode_triangles ? 3 : 1;

		size_t offset = 0;
		while (offset < mc->count)
		{
			size_t count = minSize(target_chunk, mc->count - offset);
			if (mc->mode == cgltf_meshopt_compression_mode_triangles)
				count = count / 3 * 3;

			int chunk_rc = meshopt_decodeBufferViewChunk(&assembled[offset * mc->stride], &view, offset, count);
			if (chunk_rc != 0)
			{
				printf("FAIL %s: view %zu chunk offset %zu count %zu: %d\n", path, i, offset, count, chunk_rc);
				++failures;
				break;
			}

			offset += count;
		}

		if (offset == mc->count)
		{
			if (compareBytes("bufferView chunks", assembled, full))
				++failures;
		}

		if (fallback)
		{
			std::vector<unsigned char> expected(fallback, fallback + decoded_size);
			char label[64];
			snprintf(label, sizeof(label), "fallback view %zu", i);
			if (compareFallback(label, full, expected, meshopt_BufferViewMode(mc->mode), mc->stride))
				++failures;

			meshopt_BufferView fallback_view = view;
			full.clear();
			fallback_view.data = NULL;
			fallback_view.size = 0;

			std::vector<unsigned char> fallback_chunk(minSize(target_chunk, mc->count) * mc->stride);
			rc = meshopt_decodeBufferViewChunk(&fallback_chunk[0], &fallback_view, 0, minSize(target_chunk, mc->count));
			if (rc != 0)
			{
				printf("FAIL %s: fallback-only decode %d\n", path, rc);
				++failures;
			}
			else
			{
				std::vector<unsigned char> expected_chunk(fallback, fallback + fallback_chunk.size());
				if (compareBytes("fallback chunk", fallback_chunk, expected_chunk))
					++failures;
			}
		}
	}

	printf("%s %s: %zu compressed views, %zu views with fallback, %d failures\n", failures ? "FAIL" : "PASS", path, compressed_views, fallback_views, failures);
	cgltf_free(data);
	return failures;
}

int main(int argc, char** argv)
{
	int failures = testCodecs();

	for (int i = 1; i < argc; ++i)
		failures += testGltfFile(argv[i]);

	printf("%s total failures: %d\n", failures ? "FAIL" : "PASS", failures);
	return failures ? 1 : 0;
}
