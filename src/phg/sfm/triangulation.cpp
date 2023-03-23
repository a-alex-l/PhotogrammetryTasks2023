#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    // составление однородной системы + SVD
    // без подвохов
    uint rows = 2 * count;
    uint cols = 3;

    Eigen::MatrixXd s_inv = Eigen::MatrixXd::Zero(cols, rows);
    Eigen::MatrixXd A(rows, cols);
    Eigen::VectorXd b(rows);
    for (uint i = 0; i < count; i++) {
        for (uint j = 0; j < 3; j++) {
            A(2*i, j) = ms[i][0] * Ps[i](2, j) - ms[i][2] * Ps[i](0, j);
            A(2*i+1, j) = ms[i][1] * Ps[i](2, j) - ms[i][2] * Ps[i](1, j);
        }
        b(2*i) = - ms[i][0] * Ps[i](2, 3) + ms[i][2] * Ps[i](0, 3);
        b(2*i + 1) = - ms[i][1] * Ps[i](2, 3) + ms[i][2] * Ps[i](1, 3);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    for (int i = 0; i < 3; i++)
        s_inv(i, i) = 1 / svda.singularValues()[i];

    auto ans = svda.matrixV() * s_inv * svda.matrixU().transpose() * b;
    cv::Vec4d ans_cv = {ans[0], ans[1], ans[2], 1};
    return ans_cv;
}
