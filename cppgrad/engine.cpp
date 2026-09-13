#pragma once
#include <cmath>
#include <memory>
#include <vector>
#include <set>
#include <functional>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cstdio>
#include <random>
using namespace std;

struct Value : enable_shared_from_this<Value>
{
    double data;
    double grad = 0.0;
    function<void()> _backward = []() {};
    vector<shared_ptr<Value>> _prev;
    string _operation;
    string label;

    Value(double data, vector<shared_ptr<Value>> children = {}, string operation = "", string label = "") : data(data), _prev(move(children)), _operation(move(operation)), label(move(label)) {}
};

using ValuePtr = shared_ptr<Value>;

inline ValuePtr make_value(double d, string label = "")
{
    return make_shared<Value>(d, vector<ValuePtr>{}, "", label);
}

inline ostream &operator<<(ostream &os, const ValuePtr &v)
{
    os << v->label << " Value(data) = " << v->data;
    return os;
}

//! + operation
//? For a + b
inline ValuePtr operator+(const ValuePtr &a, const ValuePtr &b)
{
    auto out = make_shared<Value>(a->data + b->data, vector<ValuePtr>{a, b}, "+");
    ValuePtr a_ = a, b_ = b, out_ = out;
    out->_backward = [a_, b_, out_]()
    {
        a_->grad += 1.0 * out_->grad;
        b_->grad += 1.0 * out_->grad;
    };
    return out;
}
//? a + 5
inline ValuePtr operator+(const ValuePtr &a, double b) { return a + make_value(b); }
//? 5 + a
inline ValuePtr operator+(double a, const ValuePtr &b) { return make_value(a) + b; }

//! * operation
//? For a * b
inline ValuePtr operator*(const ValuePtr &a, const ValuePtr &b)
{
    auto out = make_shared<Value>(a->data * b->data, vector<ValuePtr>{a, b}, "*");
    ValuePtr a_ = a, b_ = b, out_ = out;
    out->_backward = [a_, b_, out_]()
    {
        a_->grad += b_->data * out_->grad;
        b_->grad += a_->data * out_->grad;
    };
    return out;
}
//? a * 5
inline ValuePtr operator*(const ValuePtr &a, double b) { return a * make_value(b); }
//? 5 * a
inline ValuePtr operator*(double a, const ValuePtr &b) { return make_value(a) * b; }

//! pow
inline ValuePtr pow(const ValuePtr &a, double n)
{
    auto out = make_shared<Value>(std::pow(a->data, n), vector<ValuePtr>{a}, "**" + to_string(n));
    ValuePtr a_ = a, out_ = out;
    out->_backward = [a_, out_, n]()
    {
        a_->grad += n * std::pow(a_->data, n - 1) * out_->grad;
    };
    return out;
}

//! Negation
inline ValuePtr operator-(const ValuePtr &a)
{
    return a * make_value(-1);
}

//! Subtraction
//? for a - b
inline ValuePtr operator-(const ValuePtr &a, const ValuePtr &b)
{
    return a + (-b);
}
//? a - 5
inline ValuePtr operator-(const ValuePtr &a, double b) { return a - make_value(b); }
//? 5 - a
inline ValuePtr operator-(double a, const ValuePtr &b) { return make_value(a) - b; }

//! Division
//? a / b
inline ValuePtr operator/(const ValuePtr &a, const ValuePtr &b)
{
    return a * pow(b, -1);
}
//? a / 5
inline ValuePtr operator/(const ValuePtr &a, double b) { return a / make_value(b); }
//? 5 / a
inline ValuePtr operator/(double a, const ValuePtr &b) { return make_value(a) / b; }

//! Tanh
inline ValuePtr tanh(const ValuePtr &a)
{
    double x = a->data;
    double t = (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1);
    auto out = make_shared<Value>(t, vector<ValuePtr>{a}, "tanh");
    ValuePtr a_ = a, out_ = out;
    out->_backward = [a_, out_, t]()
    {
        a_->grad += (1 - t * t) * out_->grad;
    };
    return out;
}

//! exp
inline ValuePtr exp(const ValuePtr &a)
{
    auto out = make_shared<Value>(std::exp(a->data), vector<ValuePtr>{a}, "exp");
    ValuePtr a_ = a, out_ = out;
    out->_backward = [a_, out_]()
    {
        a_->grad += out_->data * out_->grad;
    };
    return out;
}

//! Back Propagation
inline void backward(const ValuePtr &root)
{
    vector<ValuePtr> topo;
    set<Value *> visited;

    function<void(const ValuePtr &)> build_topo = [&](const ValuePtr &v)
    {
        if (visited.find(v.get()) == visited.end())
        {
            visited.insert(v.get());
            for (auto &child : v->_prev)
                build_topo(child);
            topo.push_back(v);
        }
    };
    build_topo(root);

    root->grad = 1.0;
    //? reverse order calling
    for (auto it = topo.rbegin(); it != topo.rend(); ++it)
        (*it)->_backward();
}

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

inline double random_uniform(double lo = -1.0, double hi = 1.0)
{
    static mt19937 gen(random_device{}());
    uniform_real_distribution<double> dist(lo, hi);
    return dist(gen);
}

class Neuron
{
public:
    vector<ValuePtr> weight;
    ValuePtr bias;
    Neuron(int number_of_input)
    {
        for (int i = 0; i < number_of_input; i++)
            weight.push_back(make_value(random_uniform()));
        bias = make_value(random_uniform());
    }

    //? __call__ function
    ValuePtr operator()(const vector<ValuePtr> &input)
    {
        ValuePtr activation = bias;
        for (size_t i = 0; i < weight.size(); i++)
        {
            activation = activation + (weight[i] * input[i]);
        }
        return tanh(activation);
    };

    vector<ValuePtr> parameters()
    {
        vector<ValuePtr> params = weight;
        params.push_back(bias);
        return params;
    }
};

class Layer
{
public:
    vector<Neuron> neurons;

    Layer(int number_of_input, int number_of_neurons)
    {
        for (int i = 0; i < number_of_neurons; i++)
        {
            neurons.emplace_back(number_of_input);
        }
    }

    //? __call__ function
    vector<ValuePtr> operator()(const vector<ValuePtr> &x)
    {
        vector<ValuePtr> outs; //? Basically an array of tanh values from the neurons
        for (auto &n : neurons)
        {
            outs.push_back(n(x));
        }
        return outs;
    }

    vector<ValuePtr> parameters()
    {
        vector<ValuePtr> params;
        for (auto &n : neurons)
        {
            auto p = n.parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }
};

class MLP
{
public:
    vector<Layer> layers;

    MLP(int number_of_input, vector<int> &number_of_neurons_per_layer)
    {
        vector<int> network_layer_sizes;
        network_layer_sizes.push_back(number_of_input);
        for (int n : number_of_neurons_per_layer)
            network_layer_sizes.push_back(n);

        for (size_t i = 0; i < number_of_neurons_per_layer.size(); i++)
            layers.emplace_back(network_layer_sizes[i], network_layer_sizes[i + 1]);
    }

    //__call__
    vector<ValuePtr> operator()(vector<ValuePtr> x)
    {
        for (auto &layer : layers)
            x = layer(x);
        return x;
    }

    vector<ValuePtr> parameters()
    {
        vector<ValuePtr> params;
        for (auto &layer : layers)
        {
            auto p = layer.parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }
};

int main()
{
    // inputs x1, x2
    auto x1 = make_value(2.0, "x1");
    auto x2 = make_value(0.0, "x2");
    // weights w1, w2
    auto w1 = make_value(-3.0, "w1");
    auto w2 = make_value(1.0, "w2");
    // bias
    auto b = make_value(6.8813735870195432, "b");

    // x1*w1 + x2*w2 + b
    auto x1w1 = x1 * w1;
    x1w1->label = "x1w1";
    auto x2w2 = x2 * w2;
    x2w2->label = "x2w2";
    auto x1w1x2w2 = x1w1 + x2w2;
    x1w1x2w2->label = "x1w1x2w2";
    auto n = x1w1x2w2 + b;
    n->label = "n";

    auto e = exp(2.0 * n);
    e->label = "e";
    auto o = (e - 1.0) / (e + 1.0);
    o->label = "o";

    backward(o);

    draw_dot(o, "neuron_graph");

    return 0;
}