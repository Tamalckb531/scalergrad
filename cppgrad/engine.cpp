#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace std;

class Value : public enable_shared_from_this<Value>
{
public:
    double data;
    double grad = 0.0;

private:
    function<void()> _backward = []() {};
    set<shared_ptr<Value>> _prev;
    string _operation;
    string _label;

public:
    Value(
        double data,
        set<shared_ptr<Value>> children = {},
        string op = "",
        string label = "") : data(data),
                             _prev(move(children)),
                             _operation(move(op)),
                             _label(move(label)) {}

    string repr() const
    {
        return "Value(data)= " + to_string(data);
    }

    shared_ptr<Value> operator+(const shared_ptr<Value> &other)
    {
        auto out = make_shared<Value>(
            this->data + other->data,
            set<shared_ptr<Value>>{
                shared_from_this(),
                other},
            "+");
        out->_backward = [this, other, out]()
        {
            this->grad += 1.0 * out->grad;
            other->grad += 1.0 * out->grad;
        };

        return out;
    }

    shared_ptr<Value> operator+(double other)
    {
        return *this + make_shared<Value>(other);
    }
};

shared_ptr<Value> operator+(double a, const shared_ptr<Value> &b)
{
    return b->operator+(make_shared<Value>(a));
}

ostream &operator<<(ostream &os, const Value &v)
{
    return os << v.repr();
}