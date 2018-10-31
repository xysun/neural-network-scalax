// build.sc
import mill._
import mill.scalalib.scalafmt.ScalafmtModule
import scalalib._

object core extends ScalaModule with ScalafmtModule{
  def scalaVersion = "2.12.4"

  override def ivyDeps = Agg(
    ivy"co.fs2::fs2-core:1.0.0",
    ivy"co.fs2::fs2-io:1.0.0"
  )

  object test extends Tests {
    override def ivyDeps = Agg(ivy"org.scalatest::scalatest:3.0.5")
    def testFrameworks = Seq("org.scalatest.tools.Framework")
  }
}
