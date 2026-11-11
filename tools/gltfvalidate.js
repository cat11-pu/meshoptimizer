const fs = require('fs');
const path = require('path');
const validator = require(path.join(__dirname, '..', 'artifacts', 'validator', 'node_modules', 'gltf-validator'));

async function validateFile(file) {
	const bytes = new Uint8Array(fs.readFileSync(file));
	const dir = path.dirname(file);

	const result = await validator.validateBytes(bytes, {
		externalResourceFunction: uri =>
			Promise.resolve(new Uint8Array(fs.readFileSync(path.join(dir, decodeURIComponent(uri)))))
	});

	console.log(`=== ${file} ===`);
	console.log(`errors=${result.issues.numErrors} warnings=${result.issues.numWarnings} infos=${result.issues.numInfos} hints=${result.issues.numHints}`);

	for (const message of result.issues.messages) {
		console.log(`${message.severity}: ${message.code} ${message.pointer || ''} ${message.message}`);
	}

	return result.issues.numErrors;
}

(async () => {
	let errors = 0;
	for (const file of process.argv.slice(2)) {
		errors += await validateFile(file);
	}
	process.exitCode = errors ? 1 : 0;
})().catch(error => {
	console.error(error);
	process.exitCode = 1;
});
