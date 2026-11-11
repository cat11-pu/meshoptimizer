// Dumps source/plain/skinned LOD geometry as compact binary files for visualize_replay.py
#include "../src/meshoptimizer.h"
#include "skinned_meshes.h"

#include <stdio.h>
#include <string.h>

#include <vector>

static void writeDump(const char* path, const SkinnedMesh& mesh, const std::vector<unsigned int>& indices)
{
	FILE* f = fopen(path, "wb");

	if (!f)
	{
		printf("cannot open %s\n", path);
		return;
	}

	unsigned int magic = 0x534b494e; // 'SKIN'
	unsigned int vcount = (unsigned int)mesh.vertices.size();
	unsigned int icount = (unsigned int)indices.size();

	fwrite(&magic, sizeof(magic), 1, f);
	fwrite(&vcount, sizeof(vcount), 1, f);
	fwrite(&icount, sizeof(icount), 1, f);

	for (size_t i = 0; i < mesh.vertices.size(); ++i)
	{
		const SkinnedVertex& v = mesh.vertices[i];

		fwrite(&v.px, sizeof(float), 3, f);
		fwrite(v.weights, sizeof(float), 4, f);
		fwrite(v.bones, sizeof(unsigned int), 4, f);
	}

	fwrite(indices.data(), sizeof(unsigned int), indices.size(), f);
	fclose(f);
}

// writes a Wavefront OBJ of the rest-pose mesh so the generated test cases can be inspected directly
static void writeObj(const char* path, const SkinnedMesh& mesh)
{
	FILE* f = fopen(path, "w");

	if (!f)
		return;

	fprintf(f, "# %s: %zu vertices, %zu triangles; weights stored as 'w' comments per vertex\n",
	    mesh.name, mesh.vertices.size(), mesh.triangleCount());

	for (size_t i = 0; i < mesh.vertices.size(); ++i)
	{
		const SkinnedVertex& v = mesh.vertices[i];

		fprintf(f, "v %f %f %f\n", v.px, v.py, v.pz);
		fprintf(f, "# w");

		for (int k = 0; k < 4; ++k)
			if (v.weights[k] != 0.f)
				fprintf(f, " %f:b%u", v.weights[k], v.bones[k]);

		fprintf(f, "\n");
	}

	for (size_t i = 0; i < mesh.indices.size(); i += 3)
		fprintf(f, "f %u %u %u\n", mesh.indices[i] + 1, mesh.indices[i + 1] + 1, mesh.indices[i + 2] + 1);

	fclose(f);
}

int main()
{
	const float limit = 0.2f;

	std::vector<SkinnedMesh> meshes;
	meshes.push_back(makeCylinderArm());
	meshes.push_back(makeClothSheet());
	meshes.push_back(makeLegKnee());

	for (size_t m = 0; m < meshes.size(); ++m)
	{
		const SkinnedMesh& mesh = meshes[m];
		size_t target = mesh.indices.size() / 5; // aggressive LOD4-level reduction for the visual

		std::vector<unsigned int> plain(mesh.indices.size());
		size_t plain_count = meshopt_simplify(&plain[0], &mesh.indices[0], mesh.indices.size(),
		    &mesh.vertices[0].px, mesh.vertices.size(), sizeof(SkinnedVertex), target, 1e-2f, 0, NULL);
		plain.resize(plain_count);

		std::vector<float> weights(mesh.vertices.size() * 4);
		std::vector<unsigned int> bones(mesh.vertices.size() * 4);

		for (size_t i = 0; i < mesh.vertices.size(); ++i)
		{
			memcpy(&weights[i * 4], mesh.vertices[i].weights, 4 * sizeof(float));
			memcpy(&bones[i * 4], mesh.vertices[i].bones, 4 * sizeof(unsigned int));
		}

		std::vector<unsigned int> skinned(mesh.indices.size());
		std::vector<unsigned int> remap(mesh.vertices.size());
		size_t skinned_count = meshopt_simplifySkinned(&skinned[0], &mesh.indices[0], mesh.indices.size(),
		    &mesh.vertices[0].px, mesh.vertices.size(), sizeof(SkinnedVertex),
		    &weights[0], &bones[0], limit, &remap[0], target, 1e-2f, 0, NULL);
		skinned.resize(skinned_count);

		char path[256];

		snprintf(path, sizeof(path), "results/%s.source.skin", mesh.name);
		writeDump(path, mesh, mesh.indices);

		snprintf(path, sizeof(path), "models/%s.obj", mesh.name);
		writeObj(path, mesh);

		snprintf(path, sizeof(path), "results/%s.plain.skin", mesh.name);
		writeDump(path, mesh, plain);

		snprintf(path, sizeof(path), "results/%s.skinned.skin", mesh.name);
		writeDump(path, mesh, skinned);

		printf("%s: source=%zu plain=%zu skinned=%zu triangles\n",
		    mesh.name, mesh.triangleCount(), plain_count / 3, skinned_count / 3);
	}

	return 0;
}
