void foo() {}
void bar(const void* func) {}
int main() {
    bar(foo);
    return 0;
}
