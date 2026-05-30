




const path = require("path");
const { LanguageClient, TransportKind } = require("vscode-languageclient/node");

let client;

function activate(context) {
    const serverModule = path.join(__dirname, "..", "lsp_server.py");
    const serverOptions = {
        command: "python",
        args: [serverModule],
        transport: TransportKind.stdio
    };
    const clientOptions = {
        documentSelector: [{ scheme: "file", language: "triadlang" }],
    };
    client = new LanguageClient(
        "triadlang", "TriadLang Language Server",
        serverOptions, clientOptions
    );
    client.start();
}

function deactivate() {
    if (client) {
        return client.stop();
    }
}

module.exports = { activate, deactivate };
