// Procedural skinned test meshes for skinning-aware simplification validation.
// Every mesh stores float3 positions plus 4 bone weights and 4 bone indices per vertex.
#ifndef SKINNED_MESHES_H
#define SKINNED_MESHES_H

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include <vector>

struct SkinnedVertex
{
	float px, py, pz;
	float weights[4];
	unsigned int bones[4];
};

struct Vec3
{
	float x, y, z;
};

struct SkinnedMesh
{
	const char* name;
	std::vector<SkinnedVertex> vertices;
	std::vector<unsigned int> indices;
	// bone pivot points in rest pose; rotations happen around these points
	std::vector<Vec3> pivots;
	// axis-angle rotations applied per frame per bone for the animation replay test
	std::vector<Vec3> axis;
	std::vector<float> angle_amplitude;

	size_t triangleCount() const { return indices.size() / 3; }
};

static inline void setBone(SkinnedVertex& v, unsigned int bone, float weight)
{
	if (weight == 0.f)
		return;

	for (int i = 0; i < 4; ++i)
	{
		if (v.weights[i] == 0.f)
		{
			v.bones[i] = bone;
			v.weights[i] = weight;
			return;
		}
	}
}

static inline void normalizeWeights(SkinnedVertex& v)
{
	float sum = 0.f;

	for (int i = 0; i < 4; ++i)
		sum += v.weights[i];

	if (sum > 0.f)
		for (int i = 0; i < 4; ++i)
			v.weights[i] /= sum;
}

// Cylinder with two bones (0=lower, 1=upper); a sharp weight seam sits at half height.
// Models a forearm+upper-arm sleeve with an elbow boundary.
static SkinnedMesh makeCylinderArm(int segments = 48, int rows = 40)
{
	SkinnedMesh mesh;
	mesh.name = "arm_cylinder";

	const float height = 2.f;
	const float radius = 0.35f;
	const float seam = 0.5f; // normalized height of the elbow boundary

	for (int r = 0; r <= rows; ++r)
	{
		float t = float(r) / float(rows);
		float y = (t - 0.5f) * height;

		for (int s = 0; s <= segments; ++s)
		{
			float a = float(s) / float(segments) * 2.f * float(M_PI);

			SkinnedVertex v = {};
			v.px = cosf(a) * radius;
			v.py = y;
			v.pz = sinf(a) * radius;

			if (t <= seam)
			{
				setBone(v, 0, 1.f);
			}
			else
			{
				// blend band of two rows just above the seam keeps both bones, boundary below stays hard
				float blend = (t - seam) / (2.f / float(rows));
				blend = blend < 1.f ? blend : 1.f;

				setBone(v, 0, 1.f - blend);
				setBone(v, 1, blend);
				normalizeWeights(v);
			}

			mesh.vertices.push_back(v);
		}
	}

	int stride = segments + 1;

	for (int r = 0; r < rows; ++r)
		for (int s = 0; s < segments; ++s)
		{
			unsigned int a = r * stride + s;
			unsigned int b = a + 1;
			unsigned int c = a + stride;
			unsigned int d = c + 1;

			mesh.indices.push_back(a);
			mesh.indices.push_back(c);
			mesh.indices.push_back(b);
			mesh.indices.push_back(b);
			mesh.indices.push_back(c);
			mesh.indices.push_back(d);
		}

	mesh.pivots.resize(2);
	mesh.pivots[0].x = 0.f; mesh.pivots[0].y = -height * 0.5f; mesh.pivots[0].z = 0.f;
	mesh.pivots[1].x = 0.f; mesh.pivots[1].y = 0.f; mesh.pivots[1].z = 0.f;

	mesh.axis.resize(2);
	mesh.axis[0].x = 1.f; mesh.axis[0].y = 0.f; mesh.axis[0].z = 0.f;
	mesh.axis[1].x = 1.f; mesh.axis[1].y = 0.f; mesh.axis[1].z = 0.f;
	mesh.angle_amplitude.resize(2);
	mesh.angle_amplitude[0] = 0.f;
	mesh.angle_amplitude[1] = 1.1f; // elbow bends ~63 degrees

	return mesh;
}

// Flat cloth sheet with four bones arranged 2x2; vertical and horizontal seams meet at the center.
// Models a tunic/torso panel split across spine/chest/hip bones.
static SkinnedMesh makeClothSheet(int cols = 48, int rws = 48)
{
	SkinnedMesh mesh;
	mesh.name = "cloth_sheet";

	const float size = 2.f;

	for (int r = 0; r <= rws; ++r)
	{
		float v = float(r) / float(rws);

		for (int c = 0; c <= cols; ++c)
		{
			float u = float(c) / float(cols);

			SkinnedVertex vert = {};
			vert.px = (u - 0.5f) * size;
			vert.py = (v - 0.5f) * size;
			vert.pz = 0.05f * sinf(u * float(M_PI) * 2.f) * sinf(v * float(M_PI));

			// four quadrant bones (2x2); weight of the left/right bone pair and bottom/top pair
			// a narrow blend band straddles each seam; outside it each vertex binds a single bone
			float fx, fy;

			if (u < 0.45f)
				fx = 0.f;
			else if (u > 0.55f)
				fx = 1.f;
			else
				fx = (u - 0.45f) / 0.1f;

			if (v < 0.45f)
				fy = 0.f;
			else if (v > 0.55f)
				fy = 1.f;
			else
				fy = (v - 0.45f) / 0.1f;

			// bone layout: 0=BL, 1=BR, 2=TL, 3=TR
			setBone(vert, 0, (1.f - fx) * (1.f - fy));
			setBone(vert, 1, fx * (1.f - fy));
			setBone(vert, 2, (1.f - fx) * fy);
			setBone(vert, 3, fx * fy);
			normalizeWeights(vert);

			mesh.vertices.push_back(vert);
		}
	}

	int stride = cols + 1;

	for (int r = 0; r < rws; ++r)
		for (int c = 0; c < cols; ++c)
		{
			unsigned int a = r * stride + c;
			unsigned int b = a + 1;
			unsigned int d = a + stride;
			unsigned int e = d + 1;

			mesh.indices.push_back(a);
			mesh.indices.push_back(d);
			mesh.indices.push_back(b);
			mesh.indices.push_back(b);
			mesh.indices.push_back(d);
			mesh.indices.push_back(e);
		}

	mesh.pivots.resize(4);

	for (int b = 0; b < 4; ++b)
	{
		mesh.pivots[b].x = (b & 1) ? size * 0.25f : -size * 0.25f;
		mesh.pivots[b].y = (b & 2) ? size * 0.25f : -size * 0.25f;
		mesh.pivots[b].z = 0.f;
	}

	mesh.axis.resize(4);
	mesh.angle_amplitude.resize(4);

	for (int b = 0; b < 4; ++b)
	{
		mesh.axis[b].x = 0.f;
		mesh.axis[b].y = 0.f;
		mesh.axis[b].z = 1.f;
		mesh.angle_amplitude[b] = (b == 1 || b == 3) ? 0.5f : -0.5f;
	}

	return mesh;
}

// Knee-like joint: two stacked cylinders whose shared ring is a hard 2-bone seam,
// plus slight taper so geometry error stays meaningful under aggressive reduction.
static SkinnedMesh makeLegKnee(int segments = 40, int rows = 36)
{
	SkinnedMesh mesh;
	mesh.name = "leg_knee";

	const float height = 2.2f;

	for (int r = 0; r <= rows; ++r)
	{
		float t = float(r) / float(rows);
		float y = (t - 0.5f) * height;
		float taper = 1.f - 0.25f * fabsf(t - 0.5f) * 2.f;
		float radius = 0.32f * taper;

		for (int s = 0; s <= segments; ++s)
		{
			float a = float(s) / float(segments) * 2.f * float(M_PI);

			SkinnedVertex v = {};
			v.px = cosf(a) * radius;
			v.py = y;
			v.pz = sinf(a) * radius;

			if (t < 0.5f)
			{
				setBone(v, 0, 1.f);
			}
			else if (t > 0.5f)
			{
				setBone(v, 1, 1.f);
			}
			else
			{
				// seam ring itself carries both bones; neighboring rings must never merge across it
				setBone(v, 0, 0.5f);
				setBone(v, 1, 0.5f);
				normalizeWeights(v);
			}

			mesh.vertices.push_back(v);
		}
	}

	int stride = segments + 1;

	for (int r = 0; r < rows; ++r)
		for (int s = 0; s < segments; ++s)
		{
			unsigned int a = r * stride + s;
			unsigned int b = a + 1;
			unsigned int c = a + stride;
			unsigned int d = c + 1;

			mesh.indices.push_back(a);
			mesh.indices.push_back(c);
			mesh.indices.push_back(b);
			mesh.indices.push_back(b);
			mesh.indices.push_back(c);
			mesh.indices.push_back(d);
		}

	mesh.pivots.resize(2);
	mesh.pivots[0].x = 0.f; mesh.pivots[0].y = 0.f; mesh.pivots[0].z = 0.f;
	mesh.pivots[1].x = 0.f; mesh.pivots[1].y = 0.f; mesh.pivots[1].z = 0.f;

	mesh.axis.resize(2);
	mesh.axis[0].x = 0.f; mesh.axis[0].y = 0.f; mesh.axis[0].z = 1.f;
	mesh.axis[1].x = 0.f; mesh.axis[1].y = 0.f; mesh.axis[1].z = 1.f;
	mesh.angle_amplitude.resize(2);
	mesh.angle_amplitude[0] = -0.9f;
	mesh.angle_amplitude[1] = 0.9f;

	return mesh;
}

#endif
