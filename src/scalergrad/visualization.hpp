#pragma once
#include "scalergrad/engine.hpp"
#include <set>
#include <unordered_map>
#include <utility>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstdint>

inline void trace(const ValuePtr &root,
                  std::set<Value *> &nodes,
                  std::set<std::pair<Value *, Value *>> &edges,
                  std::unordered_map<Value *, ValuePtr> &keep_alive)
{
    if (nodes.find(root.get()) == nodes.end())
    {
        nodes.insert(root.get());
        keep_alive[root.get()] = root; // keep shared_ptr alive during traversal
        for (auto &child : root->_prev)
        {
            edges.insert({child.get(), root.get()});
            trace(child, nodes, edges, keep_alive);
        }
    }
}

inline void draw_dot(const ValuePtr &root, const std::string &filename = "graph")
{
    std::set<Value *> nodes;
    std::set<std::pair<Value *, Value *>> edges;
    std::unordered_map<Value *, ValuePtr> keep_alive;

    trace(root, nodes, edges, keep_alive);

    std::ostringstream dot;
    dot << "digraph G {\n";
    dot << "  rankdir=LR;\n";

    for (auto *n : nodes)
    {
        std::string uid = std::to_string(reinterpret_cast<uintptr_t>(n));

        char buf[256];
        snprintf(buf, sizeof(buf), "{%s | data %.4f | grad %.4f}",
                 n->label.c_str(), n->data, n->grad);

        dot << "  \"" << uid << "\" [label=\"" << buf << "\", shape=record];\n";

        if (!n->_operation.empty())
        {
            std::string op_uid = uid + n->_operation;
            dot << "  \"" << op_uid << "\" [label=\"" << n->_operation << "\"];\n";
            dot << "  \"" << op_uid << "\" -> \"" << uid << "\";\n";
        }
    }

    for (auto &[n1, n2] : edges)
    {
        std::string uid1 = std::to_string(reinterpret_cast<uintptr_t>(n1));
        std::string uid2 = std::to_string(reinterpret_cast<uintptr_t>(n2)) + n2->_operation;
        dot << "  \"" << uid1 << "\" -> \"" << uid2 << "\";\n";
    }

    dot << "}\n";

    std::string dot_filename = filename + ".dot";
    std::ofstream out(dot_filename);
    out << dot.str();
    out.close();

    std::cout << "DOT file written to " << dot_filename << std::endl;

    // Try to render with system Graphviz if installed
    std::string cmd = "dot -Tsvg " + dot_filename + " -o " + filename + ".svg";
    int result = std::system(cmd.c_str());
    if (result == 0)
    {
        std::cout << "SVG written to " << filename << ".svg" << std::endl;
    }
    else
    {
        std::cout << "Graphviz 'dot' not found or failed — install it "
                     "(e.g. `apt install graphviz`) and run:\n  "
                  << cmd << std::endl;
    }
}