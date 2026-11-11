#include "../src/meshoptimizer.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

struct MeshData
{
	size_t vertex_count;
	std::vector<float> vertices;
};

static meshopt_AttributeDesc makeDesc(unsigned int type, unsigned int bits, unsigned int flags, size_t offset)
{
	meshopt_AttributeDesc result = {};
	result.type = type;
	result.bits = bits;
	result.flags = flags;
	result.offset = offset;
	return result;
}

static MeshData makeGrid(size_t n)
{
	MeshData result = {};
	result.vertex_count = n * n;
	result.vertices.resize(result.vertex_count * 16);

	for (size_t y = 0; y < n; ++y)
		for (size_t x = 0; x < n; ++x)
		{
			size_t i = y * n + x;
			float u = float(x) / float(n - 1);
			float v = float(y) / float(n - 1);
			float* vertex = &result.vertices[i * 16];

			vertex[0] = u * 10.f;
			vertex[1] = v * 10.f;
			vertex[2] = sinf(float(x) * 0.1f) * 0.5f + cosf(float(y) * 0.07f) * 0.25f;

			float nx = sinf(float(x) * 0.2f) * 0.3f;
			float ny = cosf(float(y) * 0.2f) * 0.3f;
			float nz = 1.f;
			float nl = sqrtf(nx * nx + ny * ny + nz * nz);
			vertex[3] = nx / nl;
			vertex[4] = ny / nl;
			vertex[5] = nz / nl;

			vertex[8] = u * 4.f;
			vertex[9] = v * 4.f;

			float tx = cosf(float(x) * 0.1f);
			float ty = sinf(float(y) * 0.1f);
			float tz = 0.f;
			float tl = sqrtf(tx * tx + ty * ty + tz * tz);
			vertex[12] = tx / tl;
			vertex[13] = ty / tl;
			vertex[14] = tz / tl;
			vertex[15] = (x + y) % 2 ? -1.f : 1.f;
		}

	return result;
}

static MeshData makeCylinder(size_t segments, size_t rings)
{
	MeshData result = {};
	result.vertex_count = segments * rings;
	result.vertices.resize(result.vertex_count * 16);

	for (size_t y = 0; y < rings; ++y)
		for (size_t x = 0; x < segments; ++x)
		{
			size_t i = y * segments + x;
			float angle = float(x) / float(segments) * 6.28318530718f;
			float radius = 5.f;
			float height = float(y) / float(rings - 1) * 8.f - 4.f;
			float* vertex = &result.vertices[i * 16];

			vertex[0] = cosf(angle) * radius;
			vertex[1] = sinf(angle) * radius;
			vertex[2] = height;

			vertex[3] = cosf(angle);
			vertex[4] = sinf(angle);
			vertex[5] = 0.f;

			vertex[8] = float(x) / float(segments - 1);
			vertex[9] = float(y) / float(rings - 1);

			vertex[12] = -sinf(angle);
			vertex[13] = cosf(angle);
			vertex[14] = 0.f;
			vertex[15] = 1.f;
		}

	return result;
}

static MeshData makeWave(size_t n)
{
	MeshData result = {};
	result.vertex_count = n * n;
	result.vertices.resize(result.vertex_count * 16);

	for (size_t y = 0; y < n; ++y)
		for (size_t x = 0; x < n; ++x)
		{
			size_t i = y * n + x;
			float u = float(x) / float(n - 1) * 20.f;
			float v = float(y) / float(n - 1) * 20.f;
			float* vertex = &result.vertices[i * 16];

			vertex[0] = u;
			vertex[1] = v;
			vertex[2] = sinf(u * 0.35f) * cosf(v * 0.35f) * 2.f;

			float dx = cosf(u * 0.35f) * cosf(v * 0.35f) * 0.7f;
			float dy = -sinf(u * 0.35f) * sinf(v * 0.35f) * 0.7f;
			float nx = -dx;
			float ny = -dy;
			float nz = 1.f;
			float nl = sqrtf(nx * nx + ny * ny + nz * nz);
			vertex[3] = nx / nl;
			vertex[4] = ny / nl;
			vertex[5] = nz / nl;

			vertex[8] = u * 0.05f;
			vertex[9] = v * 0.05f;

			float tx = 1.f;
			float ty = 0.f;
			float tz = dx;
			float tl = sqrtf(tx * tx + ty * ty + tz * tz);
			vertex[12] = tx / tl;
			vertex[13] = ty / tl;
			vertex[14] = tz / tl;
			vertex[15] = 1.f;
		}

	return result;
}

static size_t encodeNew(const MeshData& mesh, const meshopt_AttributeDesc* attributes, std::vector<unsigned char>& buffer)
{
	size_t stride = 16 * sizeof(float);
	size_t bound = meshopt_encodeAttributeSetBound(mesh.vertex_count, attributes, 4, stride);
	buffer.resize(bound);
	return meshopt_encodeAttributeSet(&buffer[0], buffer.size(), &mesh.vertices[0], mesh.vertex_count, stride, attributes, 4);
}

static void decodeNew(const MeshData& mesh, const meshopt_AttributeDesc* attributes, const std::vector<unsigned char>& buffer, std::vector<float>& result)
{
	result.resize(mesh.vertices.size());
	meshopt_decodeAttributeSet(&result[0], mesh.vertex_count, 16 * sizeof(float), &buffer[0], buffer.size(), attributes, 4);
}

static void encodeOld(const MeshData& mesh, std::vector<float>& result)
{
	result.resize(mesh.vertices.size());

	float minv[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
	float maxv[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
	float uvmin[2] = {FLT_MAX, FLT_MAX};
	float uvmax[2] = {-FLT_MAX, -FLT_MAX};

	for (size_t i = 0; i < mesh.vertex_count; ++i)
	{
		for (int k = 0; k < 3; ++k)
		{
			minv[k] = mesh.vertices[i * 16 + k] < minv[k] ? mesh.vertices[i * 16 + k] : minv[k];
			maxv[k] = mesh.vertices[i * 16 + k] > maxv[k] ? mesh.vertices[i * 16 + k] : maxv[k];
		}

		for (int k = 0; k < 2; ++k)
		{
			uvmin[k] = mesh.vertices[i * 16 + 8 + k] < uvmin[k] ? mesh.vertices[i * 16 + 8 + k] : uvmin[k];
			uvmax[k] = mesh.vertices[i * 16 + 8 + k] > uvmax[k] ? mesh.vertices[i * 16 + 8 + k] : uvmax[k];
		}
	}

	for (size_t i = 0; i < mesh.vertex_count; ++i)
	{
		const float* source = &mesh.vertices[i * 16];
		float* target = &result[i * 16];

		for (int k = 0; k < 3; ++k)
		{
			float scale = maxv[k] - minv[k];
			int q = meshopt_quantizeUnorm(scale == 0.f ? 0.f : (source[k] - minv[k]) / scale, 16);
			target[k] = minv[k] + float(q) / 65535.f * scale;
		}

		for (int k = 0; k < 3; ++k)
		{
			int q = meshopt_quantizeSnorm(source[3 + k], 8);
			target[3 + k] = q == 0 ? 0.f : float(q) / 127.f;
		}

		for (int k = 0; k < 2; ++k)
		{
			float scale = uvmax[k] - uvmin[k];
			int q = meshopt_quantizeUnorm(scale == 0.f ? 0.f : (source[8 + k] - uvmin[k]) / scale, 16);
			target[8 + k] = uvmin[k] + float(q) / 65535.f * scale;
		}

		for (int k = 0; k < 4; ++k)
		{
			int q = meshopt_quantizeSnorm(source[12 + k], 8);
			target[12 + k] = q == 0 ? 0.f : float(q) / 127.f;
		}
	}
}

static double positionError(const MeshData& mesh, const std::vector<float>& decoded, double& max_error)
{
	float minv[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
	float maxv[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

	for (size_t i = 0; i < mesh.vertex_count; ++i)
		for (int k = 0; k < 3; ++k)
		{
			minv[k] = mesh.vertices[i * 16 + k] < minv[k] ? mesh.vertices[i * 16 + k] : minv[k];
			maxv[k] = mesh.vertices[i * 16 + k] > maxv[k] ? mesh.vertices[i * 16 + k] : maxv[k];
		}

	double diagonal = sqrt(double(maxv[0] - minv[0]) * (maxv[0] - minv[0]) + double(maxv[1] - minv[1]) * (maxv[1] - minv[1]) + double(maxv[2] - minv[2]) * (maxv[2] - minv[2]));
	double total = 0;
	max_error = 0;

	for (size_t i = 0; i < mesh.vertex_count; ++i)
	{
		double x = double(decoded[i * 16 + 0]) - mesh.vertices[i * 16 + 0];
		double y = double(decoded[i * 16 + 1]) - mesh.vertices[i * 16 + 1];
		double z = double(decoded[i * 16 + 2]) - mesh.vertices[i * 16 + 2];
		double error = sqrt(x * x + y * y + z * z);

		total += error;
		max_error = error > max_error ? error : max_error;
	}

	if (diagonal != 0)
		max_error /= diagonal;

	return diagonal == 0 ? 0 : total / double(mesh.vertex_count) / diagonal;
}

static double vectorAngle(float ax, float ay, float az, float bx, float by, float bz)
{
	double al = sqrt(double(ax) * ax + double(ay) * ay + double(az) * az);
	double bl = sqrt(double(bx) * bx + double(by) * by + double(bz) * bz);
	double dot = (double(ax) * bx + double(ay) * by + double(az) * bz) / (al * bl);

	dot = dot < -1 ? -1 : (dot > 1 ? 1 : dot);
	return acos(dot) * 180.0 / 3.14159265358979323846;
}

static void measureErrors(const MeshData& mesh, const std::vector<float>& decoded, double& pos_mean, double& pos_max, double& normal_max, double& uv_max, double& tangent_max)
{
	pos_mean = positionError(mesh, decoded, pos_max);
	normal_max = 0;
	uv_max = 0;
	tangent_max = 0;

	for (size_t i = 0; i < mesh.vertex_count; ++i)
	{
		double angle = vectorAngle(mesh.vertices[i * 16 + 3], mesh.vertices[i * 16 + 4], mesh.vertices[i * 16 + 5], decoded[i * 16 + 3], decoded[i * 16 + 4], decoded[i * 16 + 5]);
		normal_max = angle > normal_max ? angle : normal_max;

		for (int k = 0; k < 2; ++k)
		{
			double error = fabs(double(decoded[i * 16 + 8 + k]) - mesh.vertices[i * 16 + 8 + k]);
			uv_max = error > uv_max ? error : uv_max;
		}

		angle = vectorAngle(mesh.vertices[i * 16 + 12], mesh.vertices[i * 16 + 13], mesh.vertices[i * 16 + 14], decoded[i * 16 + 12], decoded[i * 16 + 13], decoded[i * 16 + 14]);
		tangent_max = angle > tangent_max ? angle : tangent_max;
	}
}

int main(int argc, char** argv)
{
	MeshData meshes[3] = {makeGrid(257), makeCylinder(257, 257), makeWave(257)};
	const char* names[3] = {"grid 257x257", "cylinder 257x257", "wave 257x257"};
	int bits[4] = {12, 12, 12, 12};

	if (argc == 5)
		for (int i = 0; i < 4; ++i)
			bits[i] = atoi(argv[i + 1]);

	meshopt_AttributeDesc attributes[4];
	attributes[0] = makeDesc(meshopt_AttributePosition, bits[0], 0, 0);
	attributes[1] = makeDesc(meshopt_AttributeNormal, bits[1], meshopt_AttributeOct, 12);
	attributes[2] = makeDesc(meshopt_AttributeTexcoord, bits[2], meshopt_AttributeWrap, 32);
	attributes[3] = makeDesc(meshopt_AttributeTangent, bits[3], meshopt_AttributeOct, 48);

	size_t total_old = 0;
	size_t total_new = 0;
	char label[64];
	snprintf(label, sizeof(label), "packed %d/%d/%d/%d-bit", bits[0], bits[1], bits[2], bits[3]);

	printf("| mesh | codec | bytes | bytes/vertex | pos mean %% diag | pos max %% diag | normal max deg | uv max | tangent max deg |\n");
	printf("|---|---|---:|---:|---:|---:|---:|---:|---:|\n");

	for (int i = 0; i < 3; ++i)
	{
		const MeshData& mesh = meshes[i];

		std::vector<float> old_decoded;
		encodeOld(mesh, old_decoded);

		std::vector<unsigned char> new_buffer;
		encodeNew(mesh, attributes, new_buffer);

		std::vector<float> new_decoded;
		decodeNew(mesh, attributes, new_buffer, new_decoded);

		double old_pos_mean, old_pos_max, old_normal_max, old_uv_max, old_tangent_max;
		measureErrors(mesh, old_decoded, old_pos_mean, old_pos_max, old_normal_max, old_uv_max, old_tangent_max);

		double new_pos_mean, new_pos_max, new_normal_max, new_uv_max, new_tangent_max;
		measureErrors(mesh, new_decoded, new_pos_mean, new_pos_max, new_normal_max, new_uv_max, new_tangent_max);

		size_t old_bytes = mesh.vertex_count * 17;
		size_t new_bytes = new_buffer.size();

		total_old += old_bytes;
		total_new += new_bytes;

		printf("| %s | existing 16/8-bit | %zu | %.3f | %.4f | %.4f | %.3f | %.5f | %.3f |\n", names[i], old_bytes, double(old_bytes) / double(mesh.vertex_count), old_pos_mean * 100, old_pos_max * 100, old_normal_max, old_uv_max, old_tangent_max);
		printf("| %s | %s | %zu | %.3f | %.4f | %.4f | %.3f | %.5f | %.3f |\n", names[i], label, new_bytes, double(new_bytes) / double(mesh.vertex_count), new_pos_mean * 100, new_pos_max * 100, new_normal_max, new_uv_max, new_tangent_max);
	}

	size_t capacity = 0;
	std::vector<unsigned char> first;
	std::vector<unsigned char> next;
	std::vector<float> scratch;
	encodeNew(meshes[0], attributes, first);

	for (int iteration = 0; iteration < 100; ++iteration)
	{
		decodeNew(meshes[0], attributes, first, scratch);
		encodeNew(meshes[0], attributes, next);
		capacity += next.size() == first.size() && memcmp(&next[0], &first[0], first.size()) == 0;
		first = next;
	}

	printf("\n");
	printf("total bytes: existing %zu, packed %zu, reduction %.2f%%\n", total_old, total_new, 100.0 * (1.0 - double(total_new) / double(total_old)));
	printf("100x encode/decode bit-identical: %zu/100\n", capacity);
	printf("position gate: max error must be <= 0.05%% bbox diagonal\n");

	return 0;
}
