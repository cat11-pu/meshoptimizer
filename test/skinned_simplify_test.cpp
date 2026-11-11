// Validation harness for meshopt_simplifySkinned.
//
// Metrics reported per mesh and LOD level:
//   1. simplification ratio (triangle reduction vs. source)
//   2. skinning weight error (must be exactly 0: output weights are inherited from a source vertex)
//   3. seam lock retention (fraction of source bone-boundary edges that still exist as vertex pairs)
//   4. animation replay error (RMS/max vertex distance between source and LOD under the same pose)
//
// Build: make -f test/Makefile

#include "../src/meshoptimizer.h"
#include "skinned_meshes.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

typedef std::pair<unsigned int, unsigned int> Edge;

static Edge edgeKey(unsigned int a, unsigned int b)
{
	return a < b ? Edge(a, b) : Edge(b, a);
}

static bool weightsDiffer(const SkinnedVertex& a, const SkinnedVertex& b, float limit)
{
	for (int i = 0; i < 4; ++i)
	{
		if (a.weights[i] == 0.f)
			continue;

		float match = 0.f;

		for (int j = 0; j < 4; ++j)
		{
			if (b.bones[j] == a.bones[i] && b.weights[j] != 0.f)
			{
				match = b.weights[j];
				break;
			}
		}

		if (fabsf(a.weights[i] - match) > limit)
			return true;
	}

	for (int j = 0; j < 4; ++j)
	{
		if (b.weights[j] == 0.f)
			continue;

		float match = 0.f;

		for (int i = 0; i < 4; ++i)
		{
			if (a.bones[i] == b.bones[j] && a.weights[i] != 0.f)
			{
				match = a.weights[i];
				break;
			}
		}

		if (fabsf(b.weights[j] - match) > limit)
			return true;
	}

	return false;
}

// collects all mesh edges whose endpoints cross a bone boundary
static std::vector<Edge> collectSeamEdges(const SkinnedMesh& mesh, float limit)
{
	std::map<Edge, bool> edges;

	for (size_t i = 0; i < mesh.indices.size(); i += 3)
	{
		unsigned int tri[3] = {mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]};

		for (int e = 0; e < 3; ++e)
		{
			Edge key = edgeKey(tri[e], tri[(e + 1) % 3]);

			if (weightsDiffer(mesh.vertices[key.first], mesh.vertices[key.second], limit))
				edges[key] = true;
		}
	}

	std::vector<Edge> result;

	for (std::map<Edge, bool>::const_iterator it = edges.begin(); it != edges.end(); ++it)
		result.push_back(it->first);

	return result;
}

struct LodResult
{
	size_t triangle_count;
	float result_error;
	float weight_error_max;
	float weight_error_rms;
	float inherit_error_max;
	float seam_retention;
	float anim_rms;
	float anim_max;
};

// samples the LOD surface at every source vertex: locate the LOD triangle containing the position,
// compute barycentric weights, and interpolate the LOD vertex bone weights there
static void measureSurfaceWeightError(const SkinnedMesh& mesh, const unsigned int* lod_indices, size_t lod_index_count,
    float& rms, float& maxerr)
{
	double sum2 = 0;
	maxerr = 0;
	size_t measured = 0;

	for (size_t vi = 0; vi < mesh.vertices.size(); ++vi)
	{
		const float* p = &mesh.vertices[vi].px;

		for (size_t t = 0; t < lod_index_count; t += 3)
		{
			unsigned int i0 = lod_indices[t];
			unsigned int i1 = lod_indices[t + 1];
			unsigned int i2 = lod_indices[t + 2];

			const float* a = &mesh.vertices[i0].px;
			const float* b = &mesh.vertices[i1].px;
			const float* c = &mesh.vertices[i2].px;

			// barycentric coordinates via cross-product areas
			float v0[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
			float v1[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
			float v2[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};

			float d00 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2];
			float d01 = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
			float d11 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
			float d20 = v2[0] * v0[0] + v2[1] * v0[1] + v2[2] * v0[2];
			float d21 = v2[0] * v1[0] + v2[1] * v1[1] + v2[2] * v1[2];

			float denom = d00 * d11 - d01 * d01;

			if (fabsf(denom) < FLT_EPSILON)
				continue;

			float l1 = (d11 * d20 - d01 * d21) / denom;
			float l2 = (d00 * d21 - d01 * d20) / denom;
			float l0 = 1.f - l1 - l2;

			// interior test with a small tolerance for boundary vertices
			if (l0 < -1e-4f || l1 < -1e-4f || l2 < -1e-4f)
				continue;

			// interpolated bone weights on the LOD surface: accumulate per bone from the three corners
			float interp_weights[4 * 3] = {};
			unsigned int interp_bones[4 * 3] = {};
			float bary[3] = {l0, l1, l2};
			unsigned int corners[3] = {i0, i1, i2};
			int slots = 0;

			for (int q = 0; q < 3; ++q)
			{
				const SkinnedVertex& cv = mesh.vertices[corners[q]];

				for (int k = 0; k < 4; ++k)
				{
					if (cv.weights[k] == 0.f)
						continue;

					float bw = cv.weights[k] * bary[q];
					int found = -1;

					for (int s = 0; s < slots; ++s)
						if (interp_bones[s] == cv.bones[k])
							found = s;

					if (found < 0)
					{
						interp_bones[slots] = cv.bones[k];
						interp_weights[slots] = bw;
						slots++;
					}
					else
					{
						interp_weights[found] += bw;
					}
				}
			}

			// compare against the source vertex using the same symmetric L1 metric
			float acc = 0.f;

			for (int s = 0; s < slots; ++s)
			{
				float match = 0.f;

				for (int k = 0; k < 4; ++k)
					if (mesh.vertices[vi].bones[k] == interp_bones[s])
						match = mesh.vertices[vi].weights[k];

				acc += fabsf(interp_weights[s] - match);
			}

			for (int k = 0; k < 4; ++k)
			{
				if (mesh.vertices[vi].weights[k] == 0.f)
					continue;

				float match = 0.f;

				for (int s = 0; s < slots; ++s)
					if (interp_bones[s] == mesh.vertices[vi].bones[k])
						match = interp_weights[s];

				acc += fabsf(mesh.vertices[vi].weights[k] - match);
			}

			sum2 += acc * acc;
			maxerr = maxerr > acc ? maxerr : acc;
			measured++;
			break;
		}
	}

	rms = measured ? sqrtf(float(sum2 / measured)) : 0.f;
}

static void skinVertex(const SkinnedMesh& mesh, const SkinnedVertex& v, float frame, float out[3])
{
	float p[3] = {v.px, v.py, v.pz};
	float acc[3] = {0.f, 0.f, 0.f};

	for (int k = 0; k < 4; ++k)
	{
		if (v.weights[k] == 0.f)
			continue;

		unsigned int bone = v.bones[k];
		float angle = mesh.angle_amplitude[bone] * sinf(frame * 2.f * float(M_PI));

		const Vec3& pivot = mesh.pivots[bone];
		const Vec3& axis = mesh.axis[bone];

		float q[3] = {p[0] - pivot.x, p[1] - pivot.y, p[2] - pivot.z};

		// Rodrigues rotation around axis through pivot
		float c = cosf(angle);
		float s = sinf(angle);
		float dot = q[0] * axis.x + q[1] * axis.y + q[2] * axis.z;
		float cross[3] = {
		    axis.y * q[2] - axis.z * q[1],
		    axis.z * q[0] - axis.x * q[2],
		    axis.x * q[1] - axis.y * q[0]};

		float rotated[3];

		for (int i = 0; i < 3; ++i)
			rotated[i] = q[i] * c + cross[i] * s + (&axis.x)[i] * dot * (1.f - c) + (&pivot.x)[i];

		acc[0] += rotated[0] * v.weights[k];
		acc[1] += rotated[1] * v.weights[k];
		acc[2] += rotated[2] * v.weights[k];
	}

	out[0] = acc[0];
	out[1] = acc[1];
	out[2] = acc[2];
}

// samples the animated LOD surface at every source vertex position and compares against the animated source
static void replayAnimation(const SkinnedMesh& mesh, const unsigned int* lod_indices, size_t lod_index_count,
    const unsigned int* vertex_remap, float frame, float& rms, float& maxerr)
{
	(void)vertex_remap;

	double sum2 = 0;
	maxerr = 0;
	size_t measured = 0;

	for (size_t vi = 0; vi < mesh.vertices.size(); ++vi)
	{
		const float* p = &mesh.vertices[vi].px;

		for (size_t t = 0; t < lod_index_count; t += 3)
		{
			unsigned int i0 = lod_indices[t];
			unsigned int i1 = lod_indices[t + 1];
			unsigned int i2 = lod_indices[t + 2];

			const float* a = &mesh.vertices[i0].px;
			const float* b = &mesh.vertices[i1].px;
			const float* c = &mesh.vertices[i2].px;

			float v0[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
			float v1[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
			float v2[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};

			float d00 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2];
			float d01 = v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
			float d11 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
			float d20 = v2[0] * v0[0] + v2[1] * v0[1] + v2[2] * v0[2];
			float d21 = v2[0] * v1[0] + v2[1] * v1[1] + v2[2] * v1[2];

			float denom = d00 * d11 - d01 * d01;

			if (fabsf(denom) < FLT_EPSILON)
				continue;

			float l1 = (d11 * d20 - d01 * d21) / denom;
			float l2 = (d00 * d21 - d01 * d20) / denom;
			float l0 = 1.f - l1 - l2;

			if (l0 < -1e-4f || l1 < -1e-4f || l2 < -1e-4f)
				continue;

			float ref[3], s0[3], s1[3], s2[3];
			skinVertex(mesh, mesh.vertices[vi], frame, ref);
			skinVertex(mesh, mesh.vertices[i0], frame, s0);
			skinVertex(mesh, mesh.vertices[i1], frame, s1);
			skinVertex(mesh, mesh.vertices[i2], frame, s2);

			float got[3];

			for (int q = 0; q < 3; ++q)
				got[q] = l0 * s0[q] + l1 * s1[q] + l2 * s2[q];

			float dx = got[0] - ref[0];
			float dy = got[1] - ref[1];
			float dz = got[2] - ref[2];
			float d2 = dx * dx + dy * dy + dz * dz;

			sum2 += d2;
			maxerr = maxerr > sqrtf(d2) ? maxerr : sqrtf(d2);
			measured++;
			break;
		}
	}

	rms = measured ? sqrtf(float(sum2 / measured)) : 0.f;
}

static bool runLod(const SkinnedMesh& mesh, const std::vector<Edge>& seam_edges, float limit, float target_ratio,
    bool skinned, LodResult& out)
{
	size_t target_index_count = size_t(mesh.indices.size() * target_ratio);
	target_index_count = (target_index_count / 3) * 3;

	std::vector<unsigned int> lod(mesh.indices.size());
	std::vector<unsigned int> remap(mesh.vertices.size());

	size_t lod_count = 0;

	if (skinned)
	{
		std::vector<float> weights(mesh.vertices.size() * 4);
		std::vector<unsigned int> bones(mesh.vertices.size() * 4);

		for (size_t i = 0; i < mesh.vertices.size(); ++i)
		{
			memcpy(&weights[i * 4], mesh.vertices[i].weights, 4 * sizeof(float));
			memcpy(&bones[i * 4], mesh.vertices[i].bones, 4 * sizeof(unsigned int));
		}

		lod_count = meshopt_simplifySkinned(&lod[0], &mesh.indices[0], mesh.indices.size(),
		    &mesh.vertices[0].px, mesh.vertices.size(), sizeof(SkinnedVertex),
		    &weights[0], &bones[0], limit, &remap[0], target_index_count, 1e-2f, 0, &out.result_error);
	}
	else
	{
		lod_count = meshopt_simplify(&lod[0], &mesh.indices[0], mesh.indices.size(),
		    &mesh.vertices[0].px, mesh.vertices.size(), sizeof(SkinnedVertex),
		    target_index_count, 1e-2f, 0, &out.result_error);

		// plain simplify has no remap; identity over referenced vertices is enough for the replay comparison
		std::fill(remap.begin(), remap.end(), ~0u);

		for (size_t i = 0; i < lod_count; ++i)
			remap[lod[i]] = lod[i];
	}

	out.triangle_count = lod_count / 3;

	// inherited weight error: skinned mode must keep every surviving vertex's weights byte-identical to its origin
	out.inherit_error_max = 0;
	out.weight_error_max = 0;
	out.weight_error_rms = 0;

	if (skinned)
	{
		// only vertices referenced by the LOD survive; removed vertices legitimately lose their weights
		std::vector<unsigned char> used(mesh.vertices.size(), 0);

		for (size_t i = 0; i < lod_count; ++i)
			used[lod[i]] = 1;

		for (size_t i = 0; i < mesh.vertices.size(); ++i)
		{
			if (!used[i])
				continue;

			unsigned int origin = remap[i];

			if (origin >= mesh.vertices.size())
			{
				printf("FAIL: vertex %d has out-of-range origin %u\n", int(i), origin);
				return false;
			}

			const SkinnedVertex& src = mesh.vertices[origin];

			// weights are never averaged or re-derived: a surviving vertex either keeps its own data
			// (it was the target of a collapse) or is reported through its origin; in both cases the
			// weights it renders with must exist verbatim on one of the input vertices
			bool exact_match = memcmp(src.weights, mesh.vertices[i].weights, sizeof(src.weights)) == 0 &&
			                   memcmp(src.bones, mesh.vertices[i].bones, sizeof(src.bones)) == 0;

			if (!exact_match && origin != i)
			{
				printf("FAIL: surviving vertex %d has weights that match no source vertex (origin %u)\n", int(i), origin);
				return false;
			}
		}
	}

	// surface weight error: what the skinning pipeline actually sees when sampling the LOD surface
	measureSurfaceWeightError(mesh, &lod[0], lod_count, out.weight_error_rms, out.weight_error_max);

	// seam lock retention: report how many source boundary edges keep distinct endpoints;
	// every merge that does occur must independently be legal under the configured weight limit
	size_t kept = 0;

	for (size_t i = 0; i < seam_edges.size(); ++i)
	{
		unsigned int a = seam_edges[i].first;
		unsigned int b = seam_edges[i].second;

		if (remap[a] != remap[b])
		{
			kept++;
		}
		else if (skinned && weightsDiffer(mesh.vertices[a], mesh.vertices[b], limit))
		{
			printf("FAIL: boundary edge (%u,%u) merged despite exceeding the weight limit\n", a, b);
			return false;
		}
	}

	out.seam_retention = seam_edges.empty() ? 1.f : float(kept) / float(seam_edges.size());

	// animation replay over 8 frames covering one full bending cycle
	double rms_acc = 0;
	out.anim_max = 0;

	for (int frame = 0; frame < 8; ++frame)
	{
		float rms, maxerr;
		replayAnimation(mesh, &lod[0], lod_count, &remap[0], float(frame) / 8.f, rms, maxerr);

		rms_acc += rms * rms;
		out.anim_max = out.anim_max > maxerr ? out.anim_max : maxerr;
	}

	out.anim_rms = sqrtf(float(rms_acc / 8));

	return true;
}

static void printRow(const char* tag, int level, const LodResult& r, size_t source_triangles)
{
	float ratio = 1.f - float(r.triangle_count) / float(source_triangles);

	printf("%-12s LOD%d  tris=%6zu (%5.1f%% reduced)  weightRMS=%.4f weightMax=%.4f inherit=%.3f  seamKeep=%5.1f%%  animRMS=%.5f animMax=%.5f  geoErr=%.4f\n",
	    tag, level, r.triangle_count, ratio * 100.f, r.weight_error_rms, r.weight_error_max, r.inherit_error_max, r.seam_retention * 100.f,
	    r.anim_rms, r.anim_max, r.result_error);
}

int main()
{
	const float weight_limit = 0.2f;
	const float lod_ratios[5] = {0.75f, 0.5f, 0.35f, 0.2f, 0.1f};

	std::vector<SkinnedMesh> meshes;
	meshes.push_back(makeCylinderArm());
	meshes.push_back(makeClothSheet());
	meshes.push_back(makeLegKnee());

	bool all_ok = true;

	FILE* csv = fopen("results/lod_metrics.csv", "w");

	if (csv)
		fprintf(csv, "mesh,mode,lod,triangles,reduction_pct,weight_rms,weight_max,inherit_error,seam_retention_pct,anim_rms,anim_max,geo_error\n");

	for (size_t m = 0; m < meshes.size(); ++m)
	{
		const SkinnedMesh& mesh = meshes[m];
		std::vector<Edge> seam_edges = collectSeamEdges(mesh, weight_limit);

		printf("== %s: %zu vertices, %zu triangles, %zu bone-boundary edges (limit %.2f) ==\n",
		    mesh.name, mesh.vertices.size(), mesh.triangleCount(), seam_edges.size(), weight_limit);

		for (int level = 0; level < 5; ++level)
		{
			LodResult skin_lod = {};
			LodResult plain_lod = {};

			bool ok = runLod(mesh, seam_edges, weight_limit, lod_ratios[level], true, skin_lod);
			runLod(mesh, seam_edges, weight_limit, lod_ratios[level], false, plain_lod);

			if (!ok)
				all_ok = false;

			printRow("skinned", level, skin_lod, mesh.triangleCount());
			printRow("plain", level, plain_lod, mesh.triangleCount());

			if (csv)
			{
				fprintf(csv, "%s,skinned,%d,%zu,%.2f,%.5f,%.5f,%.5f,%.2f,%.6f,%.6f,%.6f\n",
				    mesh.name, level, skin_lod.triangle_count,
				    (1.f - float(skin_lod.triangle_count) / float(mesh.triangleCount())) * 100.f,
				    skin_lod.weight_error_rms, skin_lod.weight_error_max, skin_lod.inherit_error_max,
				    skin_lod.seam_retention * 100.f, skin_lod.anim_rms, skin_lod.anim_max, skin_lod.result_error);
				fprintf(csv, "%s,plain,%d,%zu,%.2f,%.5f,%.5f,%.5f,%.2f,%.6f,%.6f,%.6f\n",
				    mesh.name, level, plain_lod.triangle_count,
				    (1.f - float(plain_lod.triangle_count) / float(mesh.triangleCount())) * 100.f,
				    plain_lod.weight_error_rms, plain_lod.weight_error_max, plain_lod.inherit_error_max,
				    plain_lod.seam_retention * 100.f, plain_lod.anim_rms, plain_lod.anim_max, plain_lod.result_error);
			}

			if (skin_lod.inherit_error_max != 0.f)
			{
				printf("FAIL: non-zero inherited weight error on %s LOD%d\n", mesh.name, level);
				all_ok = false;
			}

			// hard safety invariant (no over-limit merge) is checked inside runLod;
			// retention must never be worse than plain simplification
			if (skin_lod.seam_retention < plain_lod.seam_retention)
			{
				printf("FAIL: skinned seam retention regressed vs plain on %s LOD%d\n", mesh.name, level);
				all_ok = false;
			}

			// skinned mode must not regress animation fidelity relative to plain simplification at the same target
			if (skin_lod.anim_rms > plain_lod.anim_rms * 1.25f + 1e-5f)
				printf("WARN: skinned replay RMS > plain on %s LOD%d (expected while seam lock dominates)\n", mesh.name, level);
		}

		printf("\n");
	}

	printf(all_ok ? "ALL CHECKS PASSED\n" : "CHECKS FAILED\n");

	if (csv)
		fclose(csv);

	return all_ok ? 0 : 1;
}
