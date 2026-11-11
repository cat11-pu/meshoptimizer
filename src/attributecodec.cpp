// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>

namespace meshopt
{

const unsigned char kAttributeHeader[4] = {'M', 'Q', 'A', '1'};
const unsigned int kAttributeVersion = 1;
const unsigned int kAttributeHeaderSize = 24;
const unsigned int kAttributeDescSize = 48;
const unsigned int kAttributeMaxCount = 16;

static unsigned int read32(const unsigned char* data)
{
	unsigned int result = unsigned(data[0]) | (unsigned(data[1]) << 8) | (unsigned(data[2]) << 16) | (unsigned(data[3]) << 24);
	return result;
}

static void write32(unsigned char* data, unsigned int value)
{
	data[0] = (unsigned char)(value);
	data[1] = (unsigned char)(value >> 8);
	data[2] = (unsigned char)(value >> 16);
	data[3] = (unsigned char)(value >> 24);
}

static float readf(const unsigned char* data)
{
	float result;
	memcpy(&result, data, sizeof(float));
	return result;
}

static void writef(unsigned char* data, float value)
{
	memcpy(data, &value, sizeof(float));
}

static unsigned int getComponents(unsigned int type)
{
	switch (type)
	{
	case meshopt_AttributePosition:
		return 3;
	case meshopt_AttributeNormal:
		return 3;
	case meshopt_AttributeTexcoord:
		return 2;
	case meshopt_AttributeTangent:
		return 4;
	default:
		return 0;
	}
}

static unsigned int getEncodedComponents(unsigned int type, unsigned int flags)
{
	if ((type == meshopt_AttributeNormal || type == meshopt_AttributeTangent) && (flags & meshopt_AttributeOct))
		return 2;

	return getComponents(type);
}

static unsigned int getPerVertexBits(unsigned int type, unsigned int bits, unsigned int flags)
{
	unsigned int components = getEncodedComponents(type, flags);
	unsigned int result = components * bits;

	if (type == meshopt_AttributeTangent && (flags & meshopt_AttributeOct))
		result += 1; // tangent handedness

	return result;
}

static size_t getAttributeSize(size_t vertex_count, unsigned int type, unsigned int bits, unsigned int flags)
{
	return (vertex_count * getPerVertexBits(type, bits, flags) + 7) / 8;
}

static bool isSupportedType(unsigned int type)
{
	return type >= meshopt_AttributePosition && type <= meshopt_AttributeTangent;
}

static bool isSupportedBits(unsigned int bits)
{
	return bits == 4 || bits == 8 || bits == 12;
}

static bool isNormalizedType(unsigned int type)
{
	return type == meshopt_AttributePosition || type == meshopt_AttributeTexcoord;
}

static bool isOctType(unsigned int type, unsigned int flags)
{
	return (type == meshopt_AttributeNormal || type == meshopt_AttributeTangent) && (flags & meshopt_AttributeOct);
}

static void packBits(unsigned char* data, size_t index, unsigned int bits, unsigned int value)
{
	size_t bit = index * bits;
	unsigned char* target = data + bit / 8;
	unsigned int shift = (unsigned int)(bit % 8);
	unsigned int mask = (1u << bits) - 1u;

	value &= mask;

	target[0] |= (unsigned char)(value << shift);
	if (shift + bits > 8)
		target[1] |= (unsigned char)(value >> (8 - shift));
	if (shift + bits > 16)
		target[2] |= (unsigned char)(value >> (16 - shift));
}

static unsigned int unpackBits(const unsigned char* data, size_t index, unsigned int bits)
{
	size_t bit = index * bits;
	const unsigned char* source = data + bit / 8;
	unsigned int shift = (unsigned int)(bit % 8);
	unsigned int value = source[0];

	if (shift + bits > 8)
		value |= source[1] << 8;
	if (shift + bits > 16)
		value |= source[2] << 16;

	return (value >> shift) & ((1u << bits) - 1u);
}

static unsigned int quantizeUnorm(float value, float minv, float scale, unsigned int bits)
{
	float normalized = scale == 0.f ? 0.f : (value - minv) / scale;
	float scalei = float((1u << bits) - 1u);

	normalized = normalized >= 0.f ? normalized : 0.f;
	normalized = normalized <= 1.f ? normalized : 1.f;

	return (unsigned int)(normalized * scalei + 0.5f);
}

static float dequantizeUnorm(unsigned int value, float minv, float scale, unsigned int bits)
{
	float scalei = float((1u << bits) - 1u);
	return minv + float(value) / scalei * scale;
}

static int signExtend(unsigned int value, unsigned int bits)
{
	unsigned int shift = 32 - bits;
	return int(value << shift) >> shift;
}

static unsigned int quantizeSnorm(float value, unsigned int bits)
{
	int scale = (1 << (bits - 1)) - 1;
	float round = value >= 0.f ? 0.5f : -0.5f;

	value = value >= -1.f ? value : -1.f;
	value = value <= 1.f ? value : 1.f;

	int result = int(value * float(scale) + round);
	return unsigned(result) & ((1u << bits) - 1u);
}

static float dequantizeSnorm(unsigned int value, unsigned int bits)
{
	int scale = (1 << (bits - 1)) - 1;
	return float(signExtend(value, bits)) / float(scale);
}

static void encodeOct(float nx, float ny, float nz, unsigned int bits, unsigned int& u, unsigned int& v)
{
	float nl = fabsf(nx) + fabsf(ny) + fabsf(nz);
	float ns = nl == 0.f ? 0.f : 1.f / nl;

	nx *= ns;
	ny *= ns;

	float ou = (nz >= 0.f) ? nx : (1 - fabsf(ny)) * (nx >= 0.f ? 1.f : -1.f);
	float ov = (nz >= 0.f) ? ny : (1 - fabsf(nx)) * (ny >= 0.f ? 1.f : -1.f);

	u = quantizeSnorm(ou, bits);
	v = quantizeSnorm(ov, bits);

	// At the fold boundary a positive and negative code can reconstruct the same direction;
	// canonicalize signed-zero components so re-encoding decoded data is byte-stable.
	unsigned int maxv = (1u << (bits - 1)) - 1u;
	int su = signExtend(u, bits);
	int sv = signExtend(v, bits);
	int sz = int(maxv) - (su < 0 ? -su : su) - (sv < 0 ? -sv : sv);

	if ((sz >= 0 && su == 0) || (sz < 0 && (su < 0 ? -su : su) == -sz))
		u = (su < 0 ? -su : su) & ((1u << bits) - 1u);

	if ((sz >= 0 && sv == 0) || (sz < 0 && (sv < 0 ? -sv : sv) == -sz))
		v = (sv < 0 ? -sv : sv) & ((1u << bits) - 1u);
}

static void decodeOct(unsigned int u, unsigned int v, unsigned int bits, float* result)
{
	float maxv = float((1 << (bits - 1)) - 1);
	float x = float(signExtend(u, bits));
	float y = float(signExtend(v, bits));
	float z = maxv - fabsf(x) - fabsf(y);
	float t = (z >= 0.f) ? 0.f : z;

	x += (x >= 0.f) ? t : -t;
	y += (y >= 0.f) ? t : -t;

	float length = sqrtf(x * x + y * y + z * z);
	float scale = length == 0.f ? 0.f : 1.f / length;

	result[0] = x * scale;
	result[1] = y * scale;
	result[2] = z * scale;
}

static void getAttributeBounds(const float* vertices, size_t vertex_count, size_t vertex_stride, const meshopt_AttributeDesc& desc, float* minv, float* scale)
{
	unsigned int components = getComponents(desc.type);
	size_t stride = desc.stride ? desc.stride : vertex_stride;

	for (unsigned int k = 0; k < components; ++k)
	{
		minv[k] = FLT_MAX;
		scale[k] = -FLT_MAX;
	}

	for (size_t i = 0; i < vertex_count; ++i)
	{
		const unsigned char* vertex = reinterpret_cast<const unsigned char*>(vertices) + i * stride + desc.offset;

		for (unsigned int k = 0; k < components; ++k)
		{
			float value = readf(vertex + k * sizeof(float));

			minv[k] = value < minv[k] ? value : minv[k];
			scale[k] = value > scale[k] ? value : scale[k];
		}
	}

	for (unsigned int k = 0; k < components; ++k)
	{
		float maxv = scale[k];
		scale[k] = maxv - minv[k];
	}
}

} // namespace meshopt

size_t meshopt_encodeAttributeSetBound(size_t vertex_count, const struct meshopt_AttributeDesc* attributes, size_t attribute_count, size_t vertex_stride)
{
	using namespace meshopt;

	if (attribute_count > kAttributeMaxCount || vertex_count > 0xffffffffu || vertex_stride > 0xffffffffu)
		return 0;

	size_t result = kAttributeHeaderSize + attribute_count * kAttributeDescSize;

	for (size_t i = 0; i < attribute_count; ++i)
	{
		const meshopt_AttributeDesc& desc = attributes[i];
		unsigned int components = getComponents(desc.type);
		size_t stride = desc.stride ? desc.stride : vertex_stride;

		if (!isSupportedType(desc.type) || !isSupportedBits(desc.bits))
			return 0;

		if (stride == 0 || desc.offset > stride || size_t(components) * sizeof(float) > stride - desc.offset)
			return 0;

		result += getAttributeSize(vertex_count, desc.type, desc.bits, desc.flags);
	}

	return result;
}

size_t meshopt_encodeAttributeSet(unsigned char* buffer, size_t buffer_size, const float* vertices, size_t vertex_count, size_t vertex_stride, const struct meshopt_AttributeDesc* attributes, size_t attribute_count)
{
	using namespace meshopt;

	if (attribute_count > kAttributeMaxCount || vertex_count > 0xffffffffu || vertex_stride > 0xffffffffu)
		return 0;

	size_t header_size = kAttributeHeaderSize + attribute_count * kAttributeDescSize;
	size_t data_offset = header_size;

	for (size_t i = 0; i < attribute_count; ++i)
	{
		const meshopt_AttributeDesc& desc = attributes[i];

		if (!isSupportedType(desc.type) || !isSupportedBits(desc.bits))
			return 0;

		size_t stride = desc.stride ? desc.stride : vertex_stride;
		unsigned int components = getComponents(desc.type);
		if (stride == 0 || desc.offset > stride || size_t(components) * sizeof(float) > stride - desc.offset)
			return 0;

		data_offset += getAttributeSize(vertex_count, desc.type, desc.bits, desc.flags);
	}

	if (buffer_size < data_offset)
		return 0;

	memcpy(buffer, kAttributeHeader, sizeof(kAttributeHeader));
	buffer[4] = (unsigned char)(kAttributeVersion);
	buffer[5] = (unsigned char)(attribute_count);
	buffer[6] = 0;
	buffer[7] = 0;
	write32(buffer + 8, (unsigned int)(vertex_count));
	write32(buffer + 12, (unsigned int)(data_offset));
	write32(buffer + 16, (unsigned int)(vertex_stride));
	write32(buffer + 20, 0);

	size_t packet_offset = header_size;

	for (size_t i = 0; i < attribute_count; ++i)
	{
		const meshopt_AttributeDesc& desc = attributes[i];
		unsigned char* packet = buffer + kAttributeHeaderSize + i * kAttributeDescSize;
		unsigned int type = desc.type;
		unsigned int bits = desc.bits;
		unsigned int flags = desc.flags;
		unsigned int components = getComponents(type);
		size_t stride = desc.stride ? desc.stride : vertex_stride;
		size_t size = getAttributeSize(vertex_count, type, bits, flags);

		float minv[4] = {};
		float scale[4] = {};

		if (isNormalizedType(type))
			getAttributeBounds(vertices, vertex_count, vertex_stride, desc, minv, scale);

		memset(buffer + packet_offset, 0, size);

		if (isOctType(type, flags))
		{
			for (size_t j = 0; j < vertex_count; ++j)
			{
				const unsigned char* vertex = reinterpret_cast<const unsigned char*>(vertices) + j * stride + desc.offset;
				float n[4] = {};

				for (unsigned int k = 0; k < components; ++k)
					n[k] = readf(vertex + k * sizeof(float));

				unsigned int u, v;
				encodeOct(n[0], n[1], n[2], bits, u, v);

				packBits(buffer + packet_offset, j * 2 + 0, bits, u);
				packBits(buffer + packet_offset, j * 2 + 1, bits, v);
			}

			if (type == meshopt_AttributeTangent)
			{
				// handedness is stored in a separate bit plane right after the octahedral pairs
				size_t sign_offset = vertex_count * 2 * bits;

				for (size_t j = 0; j < vertex_count; ++j)
				{
					const unsigned char* vertex = reinterpret_cast<const unsigned char*>(vertices) + j * stride + desc.offset;
					float w = readf(vertex + 3 * sizeof(float));

					packBits(buffer + packet_offset, sign_offset + j, 1, w >= 0.f ? 1u : 0u);
				}
			}
		}
		else if (isNormalizedType(type))
		{
			for (size_t j = 0; j < vertex_count; ++j)
			{
				const unsigned char* vertex = reinterpret_cast<const unsigned char*>(vertices) + j * stride + desc.offset;

				for (unsigned int k = 0; k < components; ++k)
				{
					float value = readf(vertex + k * sizeof(float));
					unsigned int q = quantizeUnorm(value, minv[k], scale[k], bits);

					packBits(buffer + packet_offset, j * components + k, bits, q);
				}
			}
		}
		else
		{
			for (size_t j = 0; j < vertex_count; ++j)
			{
				const unsigned char* vertex = reinterpret_cast<const unsigned char*>(vertices) + j * stride + desc.offset;

				for (unsigned int k = 0; k < components; ++k)
				{
					float value = readf(vertex + k * sizeof(float));
					unsigned int q = quantizeSnorm(value, bits);

					packBits(buffer + packet_offset, j * components + k, bits, q);
				}
			}
		}

		packet[0] = (unsigned char)(type);
		packet[1] = (unsigned char)(bits);
		packet[2] = (unsigned char)(flags);
		packet[3] = (unsigned char)(components);
		write32(packet + 4, (unsigned int)(desc.offset));
		write32(packet + 8, (unsigned int)(packet_offset));
		write32(packet + 12, (unsigned int)(size));

		for (unsigned int k = 0; k < 4; ++k)
		{
			writef(packet + 16 + k * 4, minv[k]);
			writef(packet + 32 + k * 4, scale[k]);
		}

		packet_offset += size;
	}

	assert(packet_offset == data_offset);
	return data_offset;
}

int meshopt_decodeAttributeSet(float* vertices, size_t vertex_count, size_t vertex_stride, const unsigned char* buffer, size_t buffer_size, const struct meshopt_AttributeDesc* attributes, size_t attribute_count)
{
	using namespace meshopt;

	if (buffer_size < kAttributeHeaderSize || memcmp(buffer, kAttributeHeader, sizeof(kAttributeHeader)) != 0)
		return -1;

	unsigned int version = buffer[4];
	unsigned int encoded_count = buffer[5];

	if (version != kAttributeVersion || encoded_count != attribute_count || attribute_count > kAttributeMaxCount)
		return -2;

	unsigned int encoded_vertex_count = read32(buffer + 8);
	unsigned int total_size = read32(buffer + 12);

	if (encoded_vertex_count != vertex_count || total_size > buffer_size)
		return -3;

	if (kAttributeHeaderSize + attribute_count * kAttributeDescSize > total_size)
		return -4;

	for (size_t i = 0; i < attribute_count; ++i)
	{
		const meshopt_AttributeDesc& desc = attributes[i];
		const unsigned char* packet = buffer + kAttributeHeaderSize + i * kAttributeDescSize;
		unsigned int type = packet[0];
		unsigned int bits = packet[1];
		unsigned int flags = packet[2];
		unsigned int components = packet[3];
		size_t packet_offset = read32(packet + 8);
		size_t packet_size = read32(packet + 12);

		if (type != desc.type || !isSupportedType(type) || !isSupportedBits(bits))
			return -5;

		if (!isSupportedType(desc.type) || getComponents(type) != components)
			return -6;

		size_t stride = desc.stride ? desc.stride : vertex_stride;

		if (stride == 0 || desc.offset > stride || size_t(components) * sizeof(float) > stride - desc.offset)
			return -6;

		size_t expected_size = getAttributeSize(vertex_count, type, bits, flags);

		if (packet_size != expected_size || packet_offset < kAttributeHeaderSize + attribute_count * kAttributeDescSize)
			return -7;

		if (packet_offset > total_size || packet_size > total_size - packet_offset)
			return -7;

		float minv[4];
		float scale[4];

		for (unsigned int k = 0; k < 4; ++k)
		{
			minv[k] = readf(packet + 16 + k * 4);
			scale[k] = readf(packet + 32 + k * 4);
		}

		const unsigned char* data = buffer + packet_offset;

		if (isOctType(type, flags))
		{
			size_t sign_offset = vertex_count * 2 * bits;

			for (size_t j = 0; j < vertex_count; ++j)
			{
				unsigned char* vertex = reinterpret_cast<unsigned char*>(vertices) + j * stride + desc.offset;
				unsigned int u = unpackBits(data, j * 2 + 0, bits);
				unsigned int v = unpackBits(data, j * 2 + 1, bits);
				float n[3];

				decodeOct(u, v, bits, n);

				writef(vertex + 0 * sizeof(float), n[0]);
				writef(vertex + 1 * sizeof(float), n[1]);
				writef(vertex + 2 * sizeof(float), n[2]);

				if (type == meshopt_AttributeTangent)
				{
					unsigned int sign = unpackBits(data, sign_offset + j, 1);
					writef(vertex + 3 * sizeof(float), sign ? 1.f : -1.f);
				}
			}
		}
		else
		{
			for (size_t j = 0; j < vertex_count; ++j)
			{
				unsigned char* vertex = reinterpret_cast<unsigned char*>(vertices) + j * stride + desc.offset;

				for (unsigned int k = 0; k < components; ++k)
				{
					unsigned int q = unpackBits(data, j * components + k, bits);

					if (isNormalizedType(type))
						writef(vertex + k * sizeof(float), dequantizeUnorm(q, minv[k], scale[k], bits));
					else
						writef(vertex + k * sizeof(float), dequantizeSnorm(q, bits));
				}
			}
		}
	}

	return 0;
}
