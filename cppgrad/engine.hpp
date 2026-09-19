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
#include <vector>
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
