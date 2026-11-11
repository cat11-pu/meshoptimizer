#include "../src/meshoptimizer.h"

#define CGLTF_IMPLEMENTATION
#include "../extern/cgltf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/resource.h>
#endif

static size_t g_live = 0;
static size_t g_peak = 0;

static void* trackedAlloc(size_t size)
{
	void* result = malloc(size ? size : 1);
	if (!result)
	{
		fprintf(stderr, "out of memory allocating %zu bytes\n", size);
		exit(1);
	}

	g_live += size;
	g_peak = g_peak < g_live ? g_live : g_peak;
	return result;
}

static void trackedFree(void* data, size_t size)
{
	g_live -= size;
	free(data);
}

static unsigned int hashBytes(unsigned int hash, const unsigned char* data, size_t size)
{
	for (size_t i = 0; i < size; ++i)
	{
		hash ^= data[i];
		hash *= 16777619u;
	}

	return hash;
}

static size_t minSize(size_t a, size_t b)
{
	return a < b ? a : b;
}

static size_t getMaxRssKb()
{
#if defined(__linux__)
	FILE* file = fopen("/proc/self/status", "rb");
	if (!file)
		return 0;

	char line[256];
	size_t result = 0;

	while (fgets(line, sizeof(line), file))
	{
		if (sscanf(line, "VmHWM: %zu kB", &result) == 1)
			break;
	}

	fclose(file);
	return result;
#else
	return 0;
#endif
}

struct View
{
	meshopt_BufferView desc;
	size_t index;
};

static bool loadViews(const char* path, cgltf_data*& data, std::vector<View>& views, size_t& compressed_input, size_t& raw_total)
{
	cgltf_options options = {};

	cgltf_result result = cgltf_parse_file(&options, path, &data);
	if (result != cgltf_result_success)
	{
		fprintf(stderr, "%s: parse error %d\n", path, int(result));
		return false;
	}

	result = cgltf_load_buffers(&options, data, path);
	if (result != cgltf_result_success)
	{
		fprintf(stderr, "%s: buffer load error %d\n", path, int(result));
		cgltf_free(data);
		data = NULL;
		return false;
	}

	compressed_input = 0;
	raw_total = 0;

	for (size_t i = 0; i < data->buffer_views_count; ++i)
	{
		cgltf_buffer_view* bv = &data->buffer_views[i];
		if (!bv->has_meshopt_compression)
			continue;

		cgltf_meshopt_compression* mc = &bv->meshopt_compression;
		if (!mc->buffer->data)
		{
			fprintf(stderr, "%s: view %zu has no compressed data\n", path, i);
			cgltf_free(data);
			data = NULL;
			return false;
		}

		View view = {};
		view.index = i;
		view.desc.data = static_cast<const unsigned char*>(mc->buffer->data) + mc->offset;
		view.desc.size = mc->size;
		view.desc.fallback = bv->buffer && bv->buffer->data ? static_cast<const unsigned char*>(bv->buffer->data) + bv->offset : NULL;
		view.desc.fallback_size = view.desc.fallback ? bv->size : 0;
		view.desc.count = mc->count;
		view.desc.stride = mc->stride;
		view.desc.mode = meshopt_BufferViewMode(mc->mode);
		view.desc.filter = meshopt_BufferViewFilter(mc->filter);
		views.push_back(view);

		compressed_input += mc->size;
		raw_total += mc->count * mc->stride;
	}

	return true;
}

static size_t chunkElementCount(const View& view, size_t chunk_bytes)
{
	size_t count = chunk_bytes / view.desc.stride;
	if (count == 0)
		count = 1;

	if (view.desc.mode == meshopt_BufferViewMode_Triangles)
		count = count / 3 * 3;

	return count ? count : 3;
}

static unsigned int decodeStream(const View& view, const char* path, size_t chunk_bytes, unsigned int hash)
{
	size_t chunk_count = chunkElementCount(view, chunk_bytes);

	size_t offset = 0;
	while (offset < view.desc.count)
	{
		size_t count = minSize(chunk_count, view.desc.count - offset);
		if (view.desc.mode == meshopt_BufferViewMode_Triangles)
			count = count / 3 * 3;

		size_t decoded_size = count * view.desc.stride;
		void* decoded = trackedAlloc(decoded_size);

		int result = meshopt_decodeBufferViewChunk(decoded, &view.desc, offset, count);
		if (result != 0)
		{
			fprintf(stderr, "%s: view %zu stream decode error %d at %zu\n", path, view.index, result, offset);
			exit(1);
		}

		hash = hashBytes(hash, static_cast<unsigned char*>(decoded), decoded_size);
		trackedFree(decoded, decoded_size);

		offset += count;
	}

	return hash;
}

int main(int argc, char** argv)
{
	size_t chunk_bytes = 32 * 1024;
	const char* mode = NULL;
	const char* path = NULL;

	for (int i = 1; i < argc; ++i)
	{
		if (strncmp(argv[i], "--chunk=", 8) == 0)
			chunk_bytes = size_t(atoi(argv[i] + 8));
		else if (strncmp(argv[i], "--mode=", 7) == 0)
			mode = argv[i] + 7;
		else
			path = argv[i];
	}

	if (!path || !chunk_bytes)
	{
		fprintf(stderr, "usage: %s [--chunk=bytes] [--mode=full|stream] file.glb\n", argv[0]);
		return 1;
	}

	cgltf_data* data = NULL;
	std::vector<View> views;
	size_t compressed_input = 0;
	size_t raw_total = 0;

	if (!loadViews(path, data, views, compressed_input, raw_total))
		return 1;

	unsigned int full_hash = 2166136261u;
	unsigned int stream_hash = 2166136261u;
	size_t full_peak = 0;
	size_t stream_peak = 0;

	bool run_full = !mode || strcmp(mode, "full") == 0;
	bool run_stream = !mode || strcmp(mode, "stream") == 0;

	if (run_full)
	{
		g_live = g_peak = 0;
		std::vector<void*> keep;

		for (size_t i = 0; i < views.size(); ++i)
		{
			size_t decoded_size = views[i].desc.count * views[i].desc.stride;
			void* decoded = trackedAlloc(decoded_size);
			keep.push_back(decoded);

			int result = meshopt_decodeBufferViewChunk(decoded, &views[i].desc, 0, views[i].desc.count);
			if (result != 0)
			{
				fprintf(stderr, "%s: view %zu full decode error %d\n", path, views[i].index, result);
				return 1;
			}

			full_hash = hashBytes(full_hash, static_cast<unsigned char*>(decoded), decoded_size);
		}

		full_peak = g_peak;

		for (size_t i = 0; i < views.size(); ++i)
			trackedFree(keep[i], views[i].desc.count * views[i].desc.stride);
	}

	if (run_stream)
	{
		g_live = g_peak = 0;

		for (size_t i = 0; i < views.size(); ++i)
			stream_hash = decodeStream(views[i], path, chunk_bytes, stream_hash);

		stream_peak = g_peak;
	}

	printf("file=%s\n", path);
	printf("bufferViews=%zu compressed_input=%zu raw_total=%zu\n", views.size(), compressed_input, raw_total);

	if (run_full)
		printf("full decoded_peak=%zu total_peak=%zu\n", full_peak, compressed_input + full_peak);
	if (run_stream)
		printf("stream chunk_target=%zu decoded_peak=%zu total_peak=%zu\n", chunk_bytes, stream_peak, compressed_input + stream_peak);

	if (run_full && run_stream)
	{
		double full_total = double(compressed_input + full_peak);
		double stream_total = double(compressed_input + stream_peak);
		printf("reduction_decoded=%.2f%% reduction_total=%.2f%%\n", 100.0 - double(stream_peak) / double(full_peak) * 100.0,
		    100.0 - stream_total / full_total * 100.0);
		printf("digest_match=%s\n", full_hash == stream_hash ? "yes" : "no");
	}

	size_t rss_kb = getMaxRssKb();
	if (rss_kb)
		printf("VmHWM_kb=%zu\n", rss_kb);

	cgltf_free(data);
	return (run_full && run_stream && full_hash != stream_hash) ? 1 : 0;
}
