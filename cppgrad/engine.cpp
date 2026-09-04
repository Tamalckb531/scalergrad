#pragma once
#include <cmath>
#include <memory>
#include <vector>
#include <set>
#include <functional>
#include <iostream>
#include <string>
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

int main()
{
    auto a = make_value(5, "a");
    auto b = make_value(3, "b");

    auto c = a + b;
    c->label = "c";

    auto d = a + 5;
    d->label = "d";

    auto e = 5 + a;
    e->label = "e";

    cout << c << endl;
    cout << d << endl;
    cout << e << endl;

    return 0;
}